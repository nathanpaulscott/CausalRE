from torch import nn
from .loss_functions import down_weight_loss

class FilteringLayer(nn.Module):
    '''
    This is the trainable layer for span filtering
    it is a linear layer to reproject the hidden down down to a dimentionality of 2 (basically keep_span or throw_span)
    So think of it as an intermediate output binary classification head
    '''
    def __init__(self, hidden_size):
        super().__init__()

        self.filter_layer = nn.Linear(hidden_size, 2)

    def forward(self, embeds, label):
        '''
        Inputs are:
        - embeds are the span_reps of shape (batch, num_spans, hidden)
        - label is the span labels of shape (batch, num_spans)
        NOTE: if a span is a none-entity, the label is 0, if a span is invalid, the label is -1, other values of label >0 are positive spans

        This whole thing just passes the span reps and labels to a binary classification head and then calcs the CELoss of the labels, additionally
        they calc the difference in logits pos - neg, so this is effectively a score for how much they think it will be +ve or -ve (>0 pred is +ve etc..)
        '''
        # Extract dimensions
        B, num_spans, D = embeds.shape

        # Compute score using a predefined filtering function
        #Nathan: this is just a binary classification head reprojecting the hidden dim down to 2
        score = self.filter_layer(embeds)  # Shape: [B, num_spans, 2]

        # Modify label to binary (0 for negative class (none-entity), 1 for positive)
        #label idx = 0 is the none-entity in graphER, their documentation is shite, but it is if you trace their code
        #so here they convert the num_classes dim labels to 2, either class = 0 (negative) or class > 0 (positive)
        #so label_m is now of shape (batch, num_spans, 2), so negaitve spans and invalid spans are 0 and postive spans are 1
        label_m = label.clone()
        label_m[label_m > 0] = 1

        # Initialize the loss
        filter_loss = 0
        if self.training:
            # Compute the loss if in training mode
            #note they flatten the score and label tensors in the batch and span dim
            #and the pass sample_rate = 0.0
            #so basically this is the CRELoss of the span reps passed into a binary classification head (is span not not) vs the binary labels (a binary version of the full span labels (again, is entity or not)
            #NOTE the logits and labels are flattened and reduction is sum, so the loss output is one scalar for all spans in all obs in the batch
            filter_loss = down_weight_loss(score.view(B * num_spans, -1),
                                           label_m.view(-1),
                                           sample_rate=0.,
                                           is_logit=True)

        # Compute the filter score (difference between positive and negative class scores)
        #the score is the difference in the logits between the is_entity class and the none_entity_class 
        #so basically it ranges from -inf to +inf, >0 then the span leans towards is_entity, if < 0 it leans toward none_entity, the larger the absolute number the more certain the prediction
        filter_score = score[..., 1] - score[..., 0]  # Shape: [B, num_spans]

        # Mask out filter scores for ignored labels
        #Nathan: set the invlaid spans to -inf => no chance of being a positive case
        filter_score = filter_score.masked_fill(label == -1, float('-inf'))

        #set the positive label spans to +inf => definitely positive cases
        #Nathan: he is guaranteeing that the positive label cases end up getting choses (even if the pred incorrectly says they are not_entities)
        #this is kind of similar to what I was doing with hybrid teacher forcing, I was ensuring that all pos label spans end up making it to the span-pair stage even if being classified as none-entities, it did not work so good
        #also note that he doesn't do it for the non-training case as you have no labels so you are forced to do non-teacher forcing (which worked best even for training after a short warmup)
        if self.training:
            filter_score = filter_score.masked_fill(label_m > 0, float('inf'))

        #so the return are the scores indicating on a scale of -inf to +inf the confidence of the span being an entity with 0 being 50:50
        #the loss on the other hand is only for training and is basically the CELoss of the binary span classification head, maybe used to guage how confident the classification scores are?
        #remember filter_score is Shape: [B, num_spans] and filter_loss is a scalar
        return filter_score, filter_loss
