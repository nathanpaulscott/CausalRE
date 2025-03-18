import torch
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

    #Flatten the tensors for loss calculation
    flat_logits = logits.view(-1, logits.shape[-1])         #Flatten to (batch * num_items, num_classes)
    flat_labels = labels.view(-1).to(dtype=torch.long)      #Flatten to (batch * num_items)
    flat_mask = mask.view(-1)                               #Flatten to (batch * num_items)

    # Apply the mask by setting labels of masked-out items to -1 for ignoring
    flat_labels[~flat_mask] = -1  # Set ignored index for invalid entries

    #Calculate the loss
    loss = F.cross_entropy(flat_logits, flat_labels, reduction = reduction, ignore_index = -1)

    # Reshape loss back to the shape of labels if reduction is 'none'
    return loss.view_as(labels) if reduction == 'none' else loss



def binary_cross_entropy_loss(logits, labels, mask, reduction='sum'):
    """
    Calculates the binary cross-entropy loss for predictions versus true labels,
    considering only specified items based on a mask. Can handle both single-class
    and multi-class cases.

    Args:
    - logits (torch.Tensor): Predictions from the model, potentially shaped (batch, num_items, num_classes) or (batch, num_items).
    - labels (torch.Tensor): Binary ground truth labels, potentially shaped (batch, num_items, num_classes) or (batch, num_items).
    - mask (torch.Tensor): A boolean mask of shape (batch, num_items) that indicates which items
                           should be considered in the loss computation.
    - reduction (str): Type of loss reduction to use ('sum', 'mean', 'none'). If 'none', returns a tensor of losses.

    Returns:
    torch.Tensor: The computed loss for the valid items. If reduction is 'none', returns a tensor of individual losses.
    """
    # Determine if logits are multi-class or single-class
    if logits.dim() > 2:
        flat_logits = logits.view(-1, logits.shape[-1])
        flat_labels = labels.view(-1, labels.shape[-1]).to(dtype=flat_logits.dtype)
    else:
        flat_logits = logits.view(-1)
        flat_labels = labels.view(-1).to(dtype=flat_logits.dtype)

    flat_mask = mask.view(-1)
    selected_logits = flat_logits[flat_mask]
    selected_labels = flat_labels[flat_mask]

    # Compute the binary cross entropy loss
    loss = F.binary_cross_entropy_with_logits(selected_logits, selected_labels, reduction='none')

    # Apply reduction
    if reduction == 'sum':
        return loss.sum()
    elif reduction == 'mean':
        return loss.mean() if loss.numel() > 0 else loss.new_tensor(0.0).to(device=logits.device)  # Avoid division by zero
    elif reduction == 'none':
        return loss

    raise ValueError("Unsupported reduction type. Choose 'none', 'mean', or 'sum'.")
