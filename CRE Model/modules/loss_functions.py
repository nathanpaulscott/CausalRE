import torch
import torch.nn.functional as F

'''
This has the various loss functions used in the model
'''



def classification_loss(logits, labels, masks, reduction='sum', label_type='unilabel', class_weights=None):
    if label_type == 'unilabel':
        return cross_entropy_loss(logits, labels, masks, reduction=reduction, class_weights=class_weights)
    elif label_type == 'multilabel':
        return binary_cross_entropy_loss(logits, labels, masks, reduction=reduction, class_weights=class_weights)
    else:
        raise ValueError(f"Unsupported label type: {label_type}")
    

def process_class_weights_celoss(class_weights, flat_labels, flat_logits):
    '''
    Adjusts class weights for use in cross-entropy. Assume that each eleent in flat labels is an int.  
    0 indicates a neg case and >0 indicates a pos case
    Args:
        class_weights (str or Tensor): 'calc' to calculate weights dynamically, 'none' for no weights,
                                       or a pre-defined Tensor of weights.
        flat_labels (Tensor): The int labels tensor.
        flat_logits (Tensor): The logits tensor, used here only for shape and device information.
    
    Returns:
        Tensor or None: A tensor of class weights or None if no weighting is to be applied.
        tensor will be [1,ratio, ratio....]
    '''
    #calculate the class_weights dynamically if requested
    if class_weights == 'calc':
        #Get the ratio of pos to neg cases in labels
        count_neg_case = torch.eq(flat_labels, 0).sum().float()
        count_pos_case = torch.gt(flat_labels, 0).sum().float()
        if count_pos_case > 0:
            ratio = count_neg_case / count_pos_case
        else:
            ratio = 1  # Default ratio when there are no positive cases
        class_weights = torch.ones(flat_logits.shape[-1], dtype=flat_logits.dtype, device=flat_logits.device)
        class_weights[1:] = ratio
        return class_weights
    
    elif class_weights == 'none':
        return None
    
    else:
        return class_weights  #do nothing as class_weights will be a tensor with the static weights



def process_class_weights_bceloss(class_weights, flat_labels, flat_logits):
    """
    Adjusts class weights for use in binary cross-entropy with logits. Assumes that each
    element in flat_labels is a binary vector and any 1's indicate positive cases for respective classes.
    
    Args:
        class_weights (str or Tensor): 'calc' to calculate weights dynamically, 'none' for no weights,
                                       or a pre-defined Tensor of weights.
        flat_labels (Tensor): The binary labels tensor.
        flat_logits (Tensor): The logits tensor, used here only for shape and device information.
    
    Returns:
        Tensor or None: A tensor of class weights or None if no weighting is to be applied.  Tensor will be a single value [ratio]
    """
    if class_weights == 'calc':
        #Get the ratio of pos to neg cases in labels
        count_neg_case = (flat_labels == 0.0).sum().float()
        count_pos_case = (flat_labels == 1.0).sum().float()
        if count_pos_case > 0:
            ratio = count_neg_case / count_pos_case
        else:
            ratio = 1  # Default ratio when there are no positive cases
        class_weights = torch.tensor([ratio], dtype=flat_logits.dtype, device=flat_logits.device)
        return class_weights

    elif class_weights == 'none':
        return None

    else:
        #Return existing tensor of weights if provided directly
        return class_weights




def cross_entropy_loss(logits, labels, mask, reduction='sum', class_weights=None):
    """
    Calculates the cross-entropy loss for categorical predictions versus true labels, considering only specified items based on a mask.
    This function is for uni-label data, where each observation can only belong to one class.

    Args:
    - logits (torch.Tensor): Predictions from the model, shaped (batch, num_items, num_classes), type float.
    - labels (torch.Tensor): True labels, shaped (batch, num_items), type int64/long.
    - mask (torch.Tensor): A boolean mask shaped (batch, num_items) indicating which items should be considered.
    - reduction (str): Type of loss reduction to use ('none', 'sum', 'mean').
    - class_weights: float tensor has the weights of the class types or 'none'/None if no weights are to be used, or 'calc' if we need to calc them
    
    Returns:
    - torch.Tensor: The computed loss. Scalar if 'mean' or 'sum', tensor if 'none'.
    """
    if reduction not in ['sum', 'mean', 'none']:
        raise ValueError("Unsupported reduction type. Choose 'none', 'mean', or 'sum'.")

    #Flatten the tensors for loss calculation
    flat_logits = logits.view(-1, logits.shape[-1])  #Flatten to (batch * num_items, num_classes)
    flat_labels = labels.view(-1).to(dtype=torch.long)        #Flatten to (batch * num_items)
    flat_mask = mask.view(-1)                        #Flatten to (batch * num_items)

    # Apply the mask by setting labels of masked-out items to -1 for ignoring
    flat_labels[~flat_mask] = -1  # Set ignored index for invalid entries

    #sort out the class weights
    class_weights = process_class_weights_celoss(class_weights, flat_labels, flat_logits)

    #Calculate the loss
    loss = F.cross_entropy(flat_logits, 
                           flat_labels, 
                           reduction    = reduction, 
                           ignore_index = -1, 
                           weight       = class_weights)

    # Reshape loss back to the shape of labels if reduction is 'none'
    return loss.view_as(labels) if reduction == 'none' else loss



def binary_cross_entropy_loss(logits, labels, mask, reduction='sum', class_weights=None):
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
    - class_weights: Optional float tensor or scalar representing the weight of the positive class.

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

    # Process class weights, assume handling both binary and multi-dimensional weights
    class_weights = process_class_weights_bceloss(class_weights, selected_labels, selected_logits)

    # Compute the binary cross entropy loss
    loss = F.binary_cross_entropy_with_logits(selected_logits, selected_labels, reduction='none', pos_weight=class_weights)

    # Apply reduction
    if reduction == 'sum':
        return loss.sum()
    elif reduction == 'mean':
        return loss.mean() if loss.numel() > 0 else loss.new_tensor(0.0).to(device=logits.device)  # Avoid division by zero
    elif reduction == 'none':
        return loss

    raise ValueError("Unsupported reduction type. Choose 'none', 'mean', or 'sum'.")
