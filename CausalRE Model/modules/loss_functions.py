import torch
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
import torch.nn.functional as F

'''
This has the various loss functions used in the model
'''




def classification_loss(self, logits, labels, masks, reduction='sum', label_type='unilabel'):
    if label_type == 'unilabel':
        return cross_entropy_loss(logits, labels, masks, reduction=reduction)
    elif label_type == 'multilabel':
        return binary_cross_entropy_loss(logits, labels, masks, reduction=reduction)
    else:
        raise ValueError(f"Unsupported label type: {label_type}")
    


def cross_entropy_loss(logits, labels, mask, reduction='sum'):
    """
    Calculates the cross-entropy loss for categorical predictions versus true labels, considering only specified items based on a mask.
    This function is for uni-label data, where each observation can only belong to one class.

    Args:
    - logits (torch.Tensor): Predictions from the model, shaped (batch, num_items, num_classes), type float.
    - labels (torch.Tensor): True labels, shaped (batch, num_items), type int64/long.
    - mask (torch.Tensor): A boolean mask shaped (batch, num_items) indicating which items should be considered.
    - reduction (str): Type of loss reduction to use ('none', 'sum', 'mean').

    Returns:
    - torch.Tensor: The computed loss. Scalar if 'mean' or 'sum', tensor if 'none'.
    """
    if reduction not in ['sum', 'mean', 'none']:
        raise ValueError("Unsupported reduction type. Choose 'none', 'mean', or 'sum'.")

    #Initialize CrossEntropyLoss with internal reduction
    loss_fn = CrossEntropyLoss(reduction=reduction, ignore_index=-1)

    #Flatten the tensors for loss calculation
    flat_logits = logits.view(-1, logits.shape[-1])  #Flatten to (batch * num_items, num_classes)
    flat_labels = labels.view(-1)                    #Flatten to (batch * num_items)
    flat_mask = mask.view(-1)                        #Flatten to (batch * num_items)

    # Apply the mask by setting labels of masked-out items to -1 for ignoring
    flat_labels[~flat_mask] = -1  # Set ignored index for invalid entries

    #Calculate the loss using the internal reduction
    loss = loss_fn(flat_logits, flat_labels)

    # Reshape loss back to the shape of labels if reduction is 'none'
    return loss.view_as(labels) if reduction == 'none' else loss



def binary_cross_entropy_loss(logits, labels, mask, reduction='sum'):
    """
    Calculates the binary cross-entropy loss for predictions versus true labels in a multi-label context,
    considering only specified items based on a mask. Each label is a binary vector where multiple classes can be 1.

    Args:
    - logits (torch.Tensor): Predictions from the model, of shape (batch, num_items, num_classes), type float.
                            These should be raw logits
    - labels (torch.Tensor): Binary ground truth labels, of shape (batch, num_items, num_classes).
    - mask (torch.Tensor): A boolean mask of shape (batch, num_items) that indicates which items
                           should be considered in the loss computation.
    - reduction (str): Type of loss reduction to use ('sum', 'mean', 'none'). If 'none', a tensor of losses is returned.

    Returns:
    torch.Tensor: The computed loss for the valid items. If reduction is 'none', returns a tensor of individual losses.
    """
    if reduction not in ['sum', 'mean', 'none']:
        raise ValueError("Unsupported reduction type. Choose 'none', 'mean', or 'sum'.")

    # Flatten logits and labels to 2D (batch*num_items, num_classes)
    flat_logits = logits.view(-1, logits.shape[-1])    #Flatten to (batch * num_items, num_classes)
    flat_labels = labels.view(-1, labels.shape[-1])    #Flatten to (batch * num_items, num_classes)

    #Initialize BCEWithLogitsLoss with no reduction
    #NOTE: there is no ignore, so must do not reduction and most loss masking and reduction
    loss_fn = BCEWithLogitsLoss(reduction='none')
    #Compute the loss
    flat_loss = loss_fn(flat_logits, flat_labels)

    # Flatten the mask and expand it to apply to every class output
    flat_mask = mask.view(-1).unsqueeze(1).expand(-1, logits.shape[-1])
    # Apply the flattened mask
    masked_loss = flat_loss * flat_mask.float()

    # Perform the reduction manually
    if reduction == 'sum':
        return masked_loss.sum()
    elif reduction == 'mean':
        # Calculate mean only over the masked (non-zero) elements
        return masked_loss.sum() / flat_mask.sum()
    elif reduction == 'none':
        # Reshape the masked loss back to the original dimensions if no reduction
        return masked_loss.view_as(logits)






#This crap below here will be removed, just leave in for now and remove after you have squeezed what you can from it....


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