import torch
from torch import nn
import torch.nn.init as init

from .loss_functions import cross_entropy_loss, binary_cross_entropy_loss




class FilteringLayerBinaryDouble(nn.Module):
    """
    A binary classification head with 2 output classes (neg/pos) for determining whether to keep or discard a rep
    This layer acts as an intermediate keep head outputing two logits per rep
    which are used to calculate a filtering loss vs the binary labels and filtering score for keep/discard filtering
    NOTE: this filtering class can used as a form of smart neg sampling as opposed to the dumb random neg sampling
    The goal of neg sampling is to reduce the neg cases overwhelming the pos cases and ideally pick neg cases that are hard fo the model to learn
    This filtering algo does this well, so random neg sampling is not required in this model

    Args:
    - hidden_size (int): The size of the incoming feature dimension from the representations.
    """
    def __init__(self, hidden_size, num_limit, dropout=None):
        super().__init__()
        self.dropout = dropout
        if dropout is not None:
            self.dropout = nn.Dropout(dropout)
        self.binary_filter_head = nn.Linear(hidden_size, 2)
        #set the pos and neg num limits based on the model number preicsion
        self.pos_limit = num_limit
        self.neg_limit = -num_limit

        #self.init_weights_alt()
        self.init_weights()


    def init_weights_alt(self):
        mean, std = 0, 0.02
        init.normal_(self.binary_filter_head.weight, mean=mean, std=std)
        if self.binary_filter_head.bias is not None:
            init.constant_(self.binary_filter_head.bias, 0)


    def init_weights(self):
        init.xavier_uniform_(self.binary_filter_head.weight)
        if self.binary_filter_head.bias is not None:
            init.constant_(self.binary_filter_head.bias, 0)



    def forward(self, reps, masks, binary_labels, force_pos_cases=True, reduction='mean', loss_only=False):
        """
        Forward pass for the FilteringLayer, calculates logits, applies binary classification, computes CELoss,
        and scores the likelihood of each rep being positive or negative.

        Args:
        - reps (torch.Tensor): The representations with shape (batch, num_reps, hidden) float.
        - mask (torch.Tensor): A boolean mask with shape (batch, num_reps) bool, where True indicates a rep to use
                               and False indicates a rep to be ignored. 
                               NOTE: you need this, do not rely on the -1 encoded in the labels
        - binary_labels (torch.Tensor): binary Labels for each rep with shape (batch, num_reps) bool for unilabel and multilabel (False = neg case, True = pos case) for each rep
        - force_pos_cases: boolean flag => True means ensure pos cases are forced to be pos_limit in train mode
        - reduction (str): type of loss reduction to use 'sum'/'ave'/'none', if no reduction a tensor is returned for the loss
        - loss_only: only output loss, just output None fo filter_scores

        Returns:
        - filter_score (torch.Tensor): A tensor with shape (batch, num_reps) of scores (neg_limit to pos_limit, so they are effectively logits) representing the confidence scores of
                                      reps being positive cases. Scores range from neg_limit to pos_limit, with positive values
                                      indicating a lean towards being an pos case, and negative values the opposite.
                                      NOTE: the filter scores for pos cases are set to po_limit if teacher forcing is enabled (force_pos_cases==True)
        - filter_loss (torch.Tensor): if reduction is not 'none', then return a scalar tensor representing the cross-entropy loss over all reps if in
                                      training mode, otherwise 0.
                                      if reduction is 'none' => then a tensor of losses of shape (batch, num_reps) float
                                      NOTE: the loss calc does NOT adjust the logits to account for teacher forcing, that would mess up the training,
                                      it is simply dependent on the logits and the labels

        """
        filter_score, filter_loss = None, 0
        #Get the binary logits for each span/rel (is span/rel or not)
        if self.dropout is not None:
            reps = self.dropout(reps)
        logits = self.binary_filter_head(reps)  # Shape: (batch, num_items, 2)

        if not loss_only:
            #Compute the filter score (difference between positive and negative class logits)
            #does this for eval and training cases
            #so basically it ranges from -inf to +inf.  
            #If delta > 0 => the span/rel leans towards a pos case
            #If delta < 0 => the span/rel leans toward a neg case
            #The larger the absolute number the more certain the prediction
            filter_score = logits[..., 1] - logits[..., 0]  # Shape: [batch, num_items]
            #Mask out filter scores for maksed out labels
            #Nathan: set the masked out spans/rels to neg_limit => no chance of being a shortlisted
            filter_score = filter_score.masked_fill(~masks, self.neg_limit)
            #do final clamp to ensure all scores are with in stable limits
            filter_score = torch.clamp(filter_score, min=self.neg_limit, max=self.pos_limit)

        #Calc the filter loss (basically the CELoss for the binary labels and logits)
        #also adjust the scores to force pos cases to +inf
        #only for the case we have labels (mode in ['train', 'pred_w_labels'])
        if binary_labels is not None:
            #Compute the loss if in training mode
            #NOTE: the logits and labels are flattened and reduction is sum, so the loss output is one scalar for all spans/rels in all obs in the batch
            filter_loss = cross_entropy_loss(logits, binary_labels, masks, reduction=reduction)

            #set the positive label spans/rels to pos_limit => this is teacher forcing
            if not loss_only and force_pos_cases:
                filter_score = filter_score.masked_fill(binary_labels > 0, self.pos_limit)

        #so the return are the scores indicating on a scale of -inf to +inf the confidence of the span being an entity with 0 being 50:50
        #the loss on the other hand is only for training and is basically the CELoss of the binary span classification head, this is part of the final loss calc
        #remember filter_score is Shape: [batch, num_items] and filter_loss is a scalar or tensor depending on reduction
        return filter_score, filter_loss








class FilteringLayerBinarySingle(nn.Module):
    """
    A binary classification head with 1 output class (a score/a logit) for determining whether to keep or discard a rep
    This layer acts as an intermediate keep head outputing one logit per rep
    which are used to calculate a filtering loss vs the binary labels and filtering score for keep/discard filtering
    NOTE: this filtering class can used as a form of smart neg sampling as opposed to the dumb random neg sampling
    The goal of neg sampling is to reduce the neg cases overwhelming the pos cases and ideally pick neg cases that are hard fo the model to learn
    This filtering algo does this well, so random neg sampling is not required in this model

    Args:
    - hidden_size (int): The size of the incoming feature dimension from the representations.
    """
    def __init__(self, hidden_size, num_limit, dropout=None):
        super().__init__()
        self.dropout = dropout
        if dropout is not None:
            self.dropout = nn.Dropout(dropout)
        self.binary_filter_head = nn.Linear(hidden_size, 1)
        #set the pos and neg num limits based on the model number preicsion
        self.pos_limit = num_limit
        self.neg_limit = -num_limit

        #self.init_weights_alt()
        self.init_weights()


    def init_weights_alt(self):
        mean, std = 0, 0.02
        init.normal_(self.binary_filter_head.weight, mean=mean, std=std)
        if self.binary_filter_head.bias is not None:
            init.constant_(self.binary_filter_head.bias, 0)

    def init_weights(self):
        init.xavier_uniform_(self.binary_filter_head.weight)
        if self.binary_filter_head.bias is not None:
            init.constant_(self.binary_filter_head.bias, 0)



    def forward(self, reps, masks, binary_labels, force_pos_cases=True, reduction='mean', loss_only=False):
        """
        Forward pass for the FilteringLayer, calculates logits, applies binary classification, computes BCELoss,
        and scores the likelihood of each rep being positive or negative.

        Args:
        - reps (torch.Tensor): The representations with shape (batch, num_reps, hidden) float.
        - mask (torch.Tensor): A boolean mask with shape (batch, num_reps) bool, where True indicates a rep to use
                               and False indicates a rep to be ignored. 
                               NOTE: you need this, do not rely on the -1 encoded in the labels
        - binary_labels (torch.Tensor): binary Labels for each rep with shape (batch, num_reps) bool for unilabel and multilabel (False = neg case, True = pos case) for each rep
        - force_pos_cases: boolean flag => True means ensure pos cases are forced to be pos_limit in train mode
        - reduction (str): type of loss reduction to use 'sum'/'ave'/'none', if no reduction a tensor is returned for the loss
        - loss_only: only output loss, just output None for filter_scores

        Returns:
        - filter_score (torch.Tensor): A tensor with shape (batch, num_reps) of scores (neg_limit to pos_limit, so they are effectively logits) representing the confidence scores of
                                      reps being positive cases. Scores range from neg_limit to pos_limit, with positive values
                                      indicating a lean towards being an pos case, and negative values the opposite.
                                      NOTE: the filter scores for pos cases are set to po_limit if teacher forcing is enabled (force_pos_cases==True)
        - filter_loss (torch.Tensor): if reduction is not 'none', then return a scalar tensor representing the binary-cross-entropy loss over all reps if in
                                      training mode, otherwise 0.
                                      if reduction is 'none' => then a tensor of losses of shape (batch, num_reps) float
                                      NOTE: the loss calc does NOT adjust the logits to account for teacher forcing, that would mess up the training,
                                      it is simply dependent on the logits and the labels

        """
        filter_score, filter_loss = None, 0
        #Get the binary logits for each span/rel (is span/rel or not)
        if self.dropout is not None:
            reps = self.dropout(reps)
        logits = self.binary_filter_head(reps).squeeze(-1)  # Shape: (batch, num_items)

        if not loss_only:
            #for the single output logit case we do not have to calculate the filter_score, the logit is the filter_score
            filter_score = logits.clone()  # Shape: [batch, num_items]
            #do clamp to ensure all scores are with in stable limits
            filter_score = torch.clamp(filter_score, min=self.neg_limit, max=self.pos_limit)
            #Mask out filter scores for maksed out labels
            #Nathan: set the masked out spans/rels to neg_limit => no chance of being a shortlisted
            filter_score = filter_score.masked_fill(~masks, self.neg_limit)
        
        #Calc the filter loss (basically the BCELoss for the binary labels and logits)
        #also adjust the scores to force pos cases to +pos_limit
        #only for the case we have labels (mode in ['train', 'pred_w_labels'])
        if binary_labels is not None:
            #Compute the loss if in training mode
            #NOTE: the logits and labels are flattened and reduction is sum, so the loss output is one scalar for all spans/rels in all obs in the batch
            filter_loss = binary_cross_entropy_loss(logits, binary_labels, masks, reduction=reduction)

            #set the positive label spans/rels to pos_limit => this is teacher forcing
            if not loss_only and force_pos_cases:
                filter_score = filter_score.masked_fill(binary_labels > 0, self.pos_limit)

        #so the return are the scores indicating on a scale of -inf to +inf the confidence of the span being an entity with 0 being 50:50
        #the loss on the other hand is only for training and is basically the CELoss of the binary span classification head, this is part of the final loss calc
        #remember filter_score is Shape: [batch, num_items] and filter_loss is a scalar or tensor depending on reduction
        return filter_score, filter_loss
