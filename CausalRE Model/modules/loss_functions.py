import torch
import torch.nn.functional as F

'''
This has the various loss functions used in the model
'''

def binary_loss(preds_b, labels_b, mask, is_logit=True):
    '''
    Calculates the loss for binary span/rel predictions versus binary span/rel labels, considering only specified spans/rels based on mask.
    This function supports two types of predictions: raw logits or log softmaxed predictions. It computes loss for 
    both positive and a subset of negative samples specified by the mask.

    Parameters:
    - preds_b (torch.Tensor): Binary predictions from the model, of shape (batch, num_items, 2), type float.
                              These can be raw logits or log softmaxed outputs depending on `is_logit`.
    - labels_b (torch.Tensor): True labels, of shape (batch, num_items), type int.
    - mask (torch.Tensor): A boolean mask of shape (batch, num_items) that indicates which spans/rels
                           should be considered in the loss computation.
    - is_logit (bool): Flag indicating the type of predictions:
                       True if `preds` are raw logits (use cross_entropy loss),
                       False if `preds` are log softmaxed outputs (use nll_loss).

    Returns:
    torch.Tensor: The sum of the losses computed for both positive and the subset of negative samples.

    Usage:
    loss = loss_binary(predictions, true_labels, mask, is_logit=True)

    #################################################################################
    Nathan: this is adapted from the graphER code, I have removed almost all of their stuff, so it is completely different,
    the only thing remaining that bugs me is the reduction = 'sum', I will leave it there for now, but that may need to go also
    #################################################################################
    '''
    batch, num_items, _ = preds_b.shape

    #Determine the loss function
    loss_func = F.cross_entropy if is_logit else F.nll_loss
    #Flatten the tensors for loss
    flat_preds = preds_b.view(-1, 2)   #(batch * num_items, 2)
    flat_labels = labels_b.view(-1)    #(batch * num_items)
    flat_mask = mask.view(-1)     #(batch * num_items)
    #Masking the pos labels
    valid_pos_labels = flat_labels.clone()
    #set the ignore labels to -1
    valid_pos_labels[(flat_labels < 1) | (~flat_mask)] = -1  # Ignore
    #Masking the neg labels
    valid_neg_labels = flat_labels.clone()
    #set the ignore labels to -1
    valid_neg_labels[(flat_labels > 0) | (~flat_mask)] = -1  # Ignore
    #Calculate losses for positive and negative samples
    loss_pos = loss_func(flat_preds, valid_pos_labels, ignore_index=-1, reduction='sum')
    loss_neg = loss_func(flat_preds, valid_neg_labels, ignore_index=-1, reduction='sum')

    # Sum the losses
    return loss_pos + loss_neg





def compute_matching_loss(logits, labels, mask, num_classes):
    '''
    Have not gone through this yet, not sure what it is doing.
    '''
    B, _, _ = logits.size()

    logits_label = logits.view(-1, num_classes)
    labels = labels.view(-1)  # (batch_size * num_spans)
    mask_label = labels != -1  # (batch_size * num_spans)
    labels.masked_fill_(~mask_label, 0)  # Set the labels of padding tokens to 0

    # one-hot encoding
    labels_one_hot = torch.zeros(labels.size(0), num_classes + 1, dtype=torch.float32).to(logits.device)
    labels_one_hot.scatter_(1, labels.unsqueeze(1), 1)  # Set the corresponding index to 1
    labels_one_hot = labels_one_hot[:, 1:]  # Remove the first column

    # loss for classifier
    loss = F.binary_cross_entropy_with_logits(logits_label, labels_one_hot, reduction='none')
    # mask loss using mask (B, C)
    masked_loss = loss.view(B, -1, num_classes) * mask.unsqueeze(1)
    loss = masked_loss.view(-1, num_classes)
    # expand mask_label to loss
    mask_label = mask_label.unsqueeze(-1).expand_as(loss)
    # put lower loss for in labels_one_hot (2 for positive, 1 for negative)

    # apply mask
    loss = loss * mask_label.float()
    loss = loss.sum()

    return loss
