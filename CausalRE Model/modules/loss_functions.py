import torch
import torch.nn.functional as F

'''
This has the various loss functions used in the model
'''

def binary_loss(preds, labels, mask, is_logit=True, reduction='sum'):
    '''
    Calculates the loss for binary span/rel predictions versus binary span/rel labels, considering only specified spans/rels based on mask.
    This function supports two types of predictions: raw logits or log softmaxed predictions. It computes loss for 
    both positive and a subset of negative samples specified by the mask.

    Parameters:
    - preds_b (torch.Tensor): Binary predictions from the model, of shape (batch, num_items, 2), type float.
                              These can be raw logits or log softmaxed outputs depending on `is_logit`.
    - labels (torch.Tensor): True labels, of shape (batch, num_items), type int64/long.
    - mask (torch.Tensor): A boolean mask of shape (batch, num_items) that indicates which spans/rels
                           should be considered in the loss computation.
    - is_logit (bool): Flag indicating the type of predictions:
                       True if `preds` are raw logits (use cross_entropy loss),
                       False if `preds` are log softmaxed outputs (use nll_loss).
    - reduction (str): type of loss reduction to use 'sum'/'ave'/'none', if no reduction a tensor is returned

    Returns:
    torch.Tensor: The losses computed for the valid spans/rels.
                  If reduction is True, returns a scalar sum. Otherwise, returns a tensor of individual losses.

    Usage:
    loss = binary_loss(predictions, true_labels, mask, is_logit=True)

    #################################################################################
    Nathan: this is adapted from the graphER code, I have removed almost all of their stuff, so it is completely different,
    the only thing remaining that bugs me is the reduction = 'sum', I will leave it there for now, but that may need to go also
    #################################################################################
    '''
    batch, num_items, _ = preds.shape

    # Determine the loss function
    loss_func = F.cross_entropy
    if not is_logit:
        loss_func = F.nll_loss

    # Flatten the tensors for loss
    flat_preds = preds.view(-1, 2)   # (batch * num_items, 2)
    flat_labels = labels.view(-1)      # (batch * num_items)
    flat_mask = mask.view(-1)          # (batch * num_items)

    # Ignore invalid spans/rels using mask
    valid_labels = flat_labels
    valid_labels[~flat_mask] = -1  # Ignore invalid entries with -1

    # Calculate the loss (single pass)
    loss = loss_func(flat_preds, valid_labels, ignore_index=-1, reduction=reduction)

    #reshape the loss for the no reduction case back to (batch, num_reps)
    if reduction == 'none': 
        loss = loss.view(batch, num_items)

    return loss





def matching_loss_prompt(logits, labels, masks):
    '''
    Computes the matching loss for spans/relationships by comparing predicted logits against true labels,
    applying a binary cross-entropy loss function, and considering only the valid entries as specified by a mask.

    This function handles scenarios where the zero index represents a "negative" or non-relevant case, 
    and actual class indices start from 1. The first index is excluded from loss calculations 
    by manipulating the one-hot encoding of labels.

    Parameters:
        logits (torch.Tensor): A tensor containing the similarity scores or logits, shaped 
            (batch, top_k_spans/top_k_rels, num_types/span/rel). These scores represent the model's 
            predictions of span or relationship types for each item in a batch.
        labels (torch.Tensor): An integer tensor of true labels, shaped (batch, top_k_spans/top_k_rels),
            where each entry is an integer representing the correct class (type) of each span or relationship.
        masks (torch.Tensor): A Boolean tensor, shaped (batch, top_k_spans/top_k_rels), indicating
            valid (True) or invalid (False) entries. Only entries marked True contribute to the loss calculation.

    Returns:
        torch.Tensor: A scalar tensor representing the total loss computed across all valid entries
        in the batch, adjusted for the presence of positive and negative cases.

    Example Usage:
        # Assume logits, labels, and masks are defined with compatible shapes
        # logits: (batch_size, num_items, num_classes), labels: (batch_size, num_items),
        # masks: (batch_size, num_items) where masks indicate validity of each item
        loss = compute_matching_loss(logits, labels, masks)
    '''
    batch, num_items, num_item_types = logits.shape
    device = logits.device

    #flatten first 2 dims
    logits_flat = logits.view(-1, num_item_types) #(batch_size * num_spans, num_item_types) float
    labels_flat = labels.view(-1)  #(batch_size * num_spans) int
    masks_flat = masks.view(-1)  #(batch_size * num_spans) bool

    # Masking invalid labels directly using the provided mask
    #don't worry it will be masked out later
    labels_flat[~masks_flat] = 0  # Set the labels of non-valid items to 0
    
    # one-hot encoding
    labels_one_hot = torch.zeros(labels_flat.size(0), num_item_types + 1, dtype=torch.float32).to(device)
    labels_one_hot.scatter_(1, labels_flat.unsqueeze(1), 1)  # Set the corresponding index to 1
    labels_one_hot = labels_one_hot[:, 1:]  # Remove the first column

    #loss for classifier
    loss = F.binary_cross_entropy_with_logits(logits_flat, labels_one_hot, reduction='none')
    #apply mask to the loss
    mask_loss = masks_flat.unsqueeze(-1).expand_as(loss)
    loss = loss * mask_loss.float()  # Apply the mask to the loss
    #reduce the loss with sum
    loss = loss.sum()

    return loss





def matching_loss_no_prompt(logits, labels, masks):
    '''
    Computes the binary cross-entropy loss for a batch of items, considering only the entries 
    that are marked as valid by the item_masks.

    Parameters:
        logits (torch.Tensor): A tensor containing the logits for each item, shaped 
            (batch, num_items). These are the model's predictions of the likelihood of each item being positive.
        labels (torch.Tensor): An integer tensor of binary labels, shaped (batch, num_items),
            where each label is 0 for negative and 1 for positive cases.
        masks (torch.Tensor): A Boolean tensor, shaped (batch, num_items), indicating
            valid (True) or invalid (False) entries. Only entries marked True contribute to the loss calculation.

    Returns:
        torch.Tensor: A scalar tensor representing the total loss computed across all valid entries
        in the batch.
    '''
    # Flatten tensors to simplify the masking and loss calculation
    logits_flat = logits.view(-1)
    labels_flat = labels.view(-1)
    masks_flat = masks.view(-1)

    # Use the mask to select the valid logits and labels for the loss calculation
    valid_logits = logits_flat[masks_flat]
    valid_labels = labels_flat[masks_flat]

    # Compute the binary cross-entropy loss
    loss = F.binary_cross_entropy_with_logits(valid_logits, valid_labels.float(), reduction='sum')

    return loss



def matching_loss(logits, labels, masks, prompt=True):
    if prompt:
        return matching_loss_prompt(logits, labels, masks)
    else:
        return matching_loss_no_prompt(logits, labels, masks)