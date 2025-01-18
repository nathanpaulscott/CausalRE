import torch
from torch import nn

from .loss_functions import cross_entropy_loss




class FilteringLayer(nn.Module):
    """
    A binary classification head for determining whether to keep or discard span/rel reps.
    This layer acts as an intermediate keep head outputing two logits per rep
    which are used to calculate a filtering loss vs the binary labels and filtering score for keep/discard filtering

    Args:
    - hidden_size (int): The size of the incoming feature dimension from the representations.
    """
    def __init__(self, hidden_size, num_limit):
        super().__init__()
        self.binary_filter_head = nn.Linear(hidden_size, 2)
        #set the pos and neg num limits based on the model number preicsion
        self.pos_limit = num_limit
        self.neg_limit = -num_limit


    def apply_filter_scores_to_logits(self, logits, filter_score, thd):
        '''
        Adjusts logits based on a filter_score (-inf to inf).  
        Effectively masking out logits that fail a specified threshold. 
        This method is specifically used to do final pruning on the graph node and edge reps.
        The filter scores are converted to probs via a sigmoid function, then compared to a thd from 0 to 1
        if the prob > thd, then the span/rel is not pruned

        Args:
            logits (torch.Tensor): The original logits from the output heads.
                                   Shape should be (batch_size, top_k_spans/rels, num_classes),
                                   where num_classes depends on the specific task.
            filter_score (torch.Tensor): (-inf to inf) derived from the differenc ein binary logits from a binary head keep head
                                        indicates the likelihood that each node/edge should be kept.
                                        Shape should be (batch_size, top_k_spans/rels).
            thd (float): A threshold value (0 to 1) used to decide whether nodes or edges are kept.
                            Nodes or edges with a keep probability (sigmoid of filter_score)
                            less than this threshold are effectively masked out.

        Returns:
            torch.Tensor: The adjusted logits, where logits for nodes/edges not meeting the keep
                        threshold are set to a very large negative value (close to zero
                        influence in subsequent softmax). The shape is the same as the input logits.
        '''
        keep_prob = torch.sigmoid(filter_score)     # (batch, top_k_spans/rels)   float (between 0 and 1)
        keep_mask = (keep_prob > thd).unsqueeze(-1).float()    # (batch, top_k_spans/rels)   float (0.0 to not keep or 1.0 to keep)
        adjusted_logits = logits + (1 - keep_mask) * self.neg_limit
        return adjusted_logits




    def forward(self, reps, masks, labels_b, force_pos_cases=False, reduction='sum'):
        """
        Forward pass for the FilteringLayer, calculates logits, applies binary classification, computes CELoss,
        and scores the likelihood of each rep being positive or negative.

        Args:
        - reps (torch.Tensor): The representations with shape (batch, num_reps, hidden) float.
        - mask (torch.Tensor): A boolean mask with shape (batch, num_reps) bool, where True indicates a rep to use
                               and False indicates a rep to be ignored. 
                               NOTE: you need this, do not rely on the -1 encoded in the labels
        - labels_b (torch.Tensor): int64, binary Labels for each rep with shape (batch, num_reps) bool for unilabel and multilabel (False = neg case, True = pos case) for each rep
        - force_pos_cases: boolean flag => True means ensure pos cases are forced to be pos_limit in train mode
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
        #Get the binary logits for each span/rel (is span/rel or not)
        logits_b = self.binary_filter_head(reps)  # Shape: (batch, num_items, 2)

        #Compute the filter score (difference between positive and negative class logits)
        #does this for eval and training cases
        #so basically it ranges from -inf to +inf.  
        #If delta > 0 => the span/rel leans towards a pos case
        #If delta < 0 => the span/rel leans toward a neg case
        #The larger the absolute number the more certain the prediction
        filter_score = logits_b[..., 1] - logits_b[..., 0]  # Shape: [batch, num_items]

        #Mask out filter scores for maksed out labels
        #Nathan: set the masked out spans/rels to neg_limit => no chance of being a positive case
        filter_score = filter_score.masked_fill(~masks, self.neg_limit)
        
        #do final clamp to ensure all scores are with in stable limits
        filter_score = torch.clamp(filter_score, min=self.neg_limit, max=self.pos_limit)

        #Calc the filter loss (basically the CELoss for the binary labels and logits)
        #also adjust the scores to force pos cases to +inf
        #only for the case we have labels (mode in ['train', 'pred_w_labels'])
        filter_loss = 0
        if labels_b is not None:
            #set the positive label spans/rels to pos_limit => force_pos_cases
            #this is a form of teacher forcing, we are guaranteeing that positive span cases make it to the initial graph
            #I put in code to be able to turn this off and also to turn it off after a set number of batches after the model has honed in on a good state (this is what worked best for me in other models)
            if force_pos_cases:
                filter_score = filter_score.masked_fill(labels_b > 0, self.pos_limit)
                
                #FOR TESTING, to force pos to NOT be in the top_k
                #filter_score = filter_score.masked_fill(labels_b > 0, self.neg_limit)
                #FOR TESTING

            #Compute the loss if in training mode
            #NOTE: the logits and labels are flattened and reduction is sum, so the loss output is one scalar for all spans/rels in all obs in the batch
            filter_loss = cross_entropy_loss(logits_b, labels_b, masks, reduction=reduction)

        #so the return are the scores indicating on a scale of -inf to +inf the confidence of the span being an entity with 0 being 50:50
        #the loss on the other hand is only for training and is basically the CELoss of the binary span classification head, this is part of the final loss calc
        #remember filter_score is Shape: [batch, num_items] and filter_loss is a scalar or tensor depending on reduction
        return filter_score, filter_loss
