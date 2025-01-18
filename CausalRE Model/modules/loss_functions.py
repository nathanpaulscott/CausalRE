import torch
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
import torch.nn.functional as F

'''
This has the various loss functions used in the model
'''




def classification_loss(logits, labels, masks, reduction='sum', label_type='unilabel'):
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

    #Flatten logits to 2D (batch*num_items, num_classes)
    flat_logits = logits.view(-1, logits.shape[-1])    #Flatten to (batch * num_items, num_classes)
    #Flatten labels to 2D (batch*num_items, num_classes) and set to the same float type as the logits
    flat_labels = labels.view(-1, labels.shape[-1]).to(dtype=flat_logits.dtype)    #Flatten to (batch * num_items, num_classes)
    
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




