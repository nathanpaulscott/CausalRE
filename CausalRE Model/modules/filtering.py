import torch
from torch import nn
from .loss_functions import binary_loss

class FilteringLayer(nn.Module):
    """
    A binary classification head for determining whether to keep or discard reps.
    This layer acts as an intermediate keep head outputing two logits per rep
    which are used to calculate a filtering loss vs the labels and filtering score for keep/discard filtering

    Args:
    - hidden_size (int): The size of the incoming feature dimension from the representations.
    """
    def __init__(self, hidden_size):
        super().__init__()
        self.binary_filter_head = nn.Linear(hidden_size, 2)


    def forward(self, reps, masks, labels, force_pos_cases=False, reduction='sum'):
        """
        Forward pass for the FilteringLayer, calculates logits, applies binary classification, computes loss,
        and scores the likelihood of each rep being positive or negative.

        Args:
        - reps (torch.Tensor): The representations with shape (batch, num_reps, hidden) float.
        - mask (torch.Tensor): A boolean mask with shape (batch, num_reps) bool, where True indicates a rep to use
                               and False indicates a rep to be ignored. 
                               NOTE: you need this, do not rely on the -1 encoded in the labels
        - labels (torch.Tensor): Labels for each rep with shape (batch, num_reps) int. Labels are 0 for
                                 negative-cases, potentially -1 for invalids, and positive integers for positive cases.
        - force_pos_cases: boolean flag => True means ensure pos cases are forced to be +inf in train mode
        - reduction (str): type of loss reduction to use 'sum'/'ave'/'none', if no reduction a tensor is returned for the loss
        
        Returns:
        - filter_score (torch.Tensor): A tensor with shape (batch, num_reps) representing the confidence scores of
                                      reps being positive cases. Scores range from -inf to +inf, with positive values
                                      indicating a lean towards being an pos case, and negative values the opposite.
        - filter_loss (torch.Tensor): if reduction is not 'none', then return a scalar tensor representing the cross-entropy loss over all reps if in
                                      training mode, otherwise 0.
                                      if reduction is 'none' => then a tensor of losses of shape (batch, num_reps) float

        Notes:
        During training, if force_pos is True, positive labeled reps are forced to be selected by setting their scores to +inf, ensuring
        their selection despite potential misclassifications by the logits. This method mirrors certain teacher-forcing
        techniques used in training to guide model behavior.
        """
        #Extract dimensions
        batch, num_items, hidden = reps.shape

        #Get the binary logits for each span/rel (is span/rel or not)
        logits_b = self.binary_filter_head(reps)  # Shape: (batch, num_items, 2)

        #Make the binary int labels from labels
        labels_b = (labels > 0).to(torch.int64)

        #Calc the filter loss (basically the CELoss for the binary labels and logits)
        #only for the training case
        filter_loss = 0
        if self.training:
            #Compute the loss if in training mode
            #NOTE the logits and labels are flattened and reduction is sum, so the loss output is one scalar for all spans/rels in all obs in the batch
            filter_loss = binary_loss(logits_b, labels_b, masks, is_logit=True, reduction=reduction)

        #Compute the filter score (difference between positive and negative class logits)
        #does this for eval and training cases
        #so basically it ranges from -inf to +inf.  
        #If delta > 0 => the span/rel leans towards a pos case
        #If delta < 0 => the span/rel leans toward a neg case
        #The larger the absolute number the more certain the prediction
        filter_score = logits_b[..., 1] - logits_b[..., 0]  # Shape: [batch, num_items]

        #Mask out filter scores for maksed out labels
        #Nathan: set the masked out spans/rels to -inf => no chance of being a positive case
        filter_score = filter_score.masked_fill(~masks, float('-inf'))
        
        #set the positive label spans/rels to +inf => definitely positive cases
        #this is a form of teacher forcing, we are guaranteeing that positive span cases make it to the initial graph
        #I put in code to be able to turn this off and also to turn it off after a set number of batches after the model has honed in on a good state (this is what worked best for me in other models)
        if self.training and force_pos_cases:
            filter_score = filter_score.masked_fill(labels_b > 0, float('inf'))

        #so the return are the scores indicating on a scale of -inf to +inf the confidence of the span being an entity with 0 being 50:50
        #the loss on the other hand is only for training and is basically the CELoss of the binary span classification head, this is part of the final loss calc
        #remember filter_score is Shape: [batch, num_items] and filter_loss is a scalar or tensor depending on reduction
        return filter_score, filter_loss
