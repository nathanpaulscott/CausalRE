from torch import nn
from .loss_functions import binary_loss

class FilteringLayer(nn.Module):
    """
    A trainable classification head for determining whether to keep or discard a spans/rels based their representations.
    This layer acts as an intermediate binary classification head outputting two logits per span/rel, which are used
    to calculate a "keep span/rel" or "discard span/rel" decision.

    Args:
    - hidden_size (int): The size of the incoming feature dimension from the span/rel representations.
    """
    def __init__(self, hidden_size):
        super().__init__()
        self.binary_filter_head = nn.Linear(hidden_size, 2)



    def forward(self, reps, mask, labels, force_pos_cases=False):
        """
        Forward pass for the FilteringLayer, calculates logits, applies binary classification, computes loss,
        and scores the likelihood of each span/rel being positive or negative.

        Args:
        - reps (torch.Tensor): The span/rel representations with shape (batch, num_spans/rels, hidden_size).
        - mask (torch.Tensor): A boolean mask with shape (batch, num_spans/rels) where True indicates a valid span/rel to use
                               and False indicates a span/rel to be ignored. => you need this, do not rely on teh -1 encoded in the labels
        - labels (torch.Tensor): Labels for each span/rel with shape (batch, num_spans/rels). Labels are 0 for
                                 none-span/rel, potentially -1 for invalid spans/rels, and positive integers for valid spans/rels.
        - force_pos_cases: boolean flag => True menas ensure pos cases are forced to be +inf in train mode, do for spans, not for rels
        
        Returns:
        - filter_score (torch.Tensor): A tensor with shape [B, num_spans/rels] representing the confidence scores of
                                      spans/rels being positive cases. Scores range from -inf to +inf, with positive values
                                      indicating a lean towards being an pos case, and negative values the opposite.
        - filter_loss (torch.Tensor): A scalar tensor representing the cross-entropy loss over all spans/rels if in
                                      training mode, otherwise 0.

        Notes:
        During training, positive labeled spans/rels are forced to be selected by setting their scores to +inf, ensuring
        their selection despite potential misclassifications by the logits. This method mirrors certain teacher-forcing
        techniques used in training to guide model behavior.
        """
        #Extract dimensions
        B, num_items, D = reps.shape

        #Get the binary logits for each span/rel (is span/rel or not)
        logits_b = self.binary_filter_head(reps)  # Shape: [B, num_items, 2]

        #Make the binary span/rel labels from labels
        labels_b = labels.clone()
        labels_b[labels_b > 0] = 1

        #Calc the filter loss (basically the CELoss for the binary labels and logits)
        #only for the training case
        filter_loss = 0
        if self.training:
            #Compute the loss if in training mode
            #NOTE the logits and labels are flattened and reduction is sum, so the loss output is one scalar for all spans/rels in all obs in the batch
            filter_loss = binary_loss(logits_b, labels_b, mask, is_logit=True)

        #Compute the filter score (difference between positive and negative class logits)
        #does this for eval and training cases
        #so basically it ranges from -inf to +inf.  
        #If delta > 0 => the span/rel leans towards a pos case
        #If delta < 0 => the span/rel leans toward a neg case
        #The larger the absolute number the more certain the prediction
        filter_score = logits_b[..., 1] - logits_b[..., 0]  # Shape: [B, num_items]

        #Mask out filter scores for maksed out labels
        #Nathan: set the masked out spans/rels to -inf => no chance of being a positive case
        filter_score = filter_score.masked_fill(~mask, float('-inf'))
        
        #set the positive label spans/rels to +inf => definitely positive cases
        #Nathan: he is guaranteeing that the positive label cases end up getting chosen (even if the pred incorrectly says they are not_entities)
        #this is kind of similar to what I was doing with hybrid teacher forcing, I was ensuring that all pos label spans/rels end up making it to the span-pair stage even if being classified as none-entities, it did not work so good
        #also note that he doesn't do it for the non-training case as you have no labels so you are forced to do non-teacher forcing (which worked best even for training after a short warmup)
        if self.training and force_pos_cases:
            filter_score = filter_score.masked_fill(labels_b > 0, float('inf'))

        #so the return are the scores indicating on a scale of -inf to +inf the confidence of the span being an entity with 0 being 50:50
        #the loss on the other hand is only for training and is basically the CELoss of the binary span classification head, maybe used to guage how confident the classification scores are?
        #remember filter_score is Shape: [B, num_spans] and filter_loss is a scalar
        return filter_score, filter_loss
