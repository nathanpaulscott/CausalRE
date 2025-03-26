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
    loss = F.cross_entropy(flat_logits, flat_labels, reduction=reduction, ignore_index=-1)

    ''''
    if torch.isnan(loss).any():
        print('xxxxxxxxxxxxxxxxxxxxxxx')
        print(logits)
        print(labels)
        print(mask)
        exit()
    '''

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





def cosine_similarity_loss(pred_reps, gold_reps, neg_limit, pred_labels=None, gold_labels=None, pred_masks=None, gold_masks=None):
    """
    Computes a cosine similarity-based loss between predicted and gold representations,
    considering matches between entries with the same label. If labels are not provided,
    the function defaults to calculating similarity without label matching, effectively performing
    a binary comparison across all pairs.

    Args:
        pred_reps (Tensor): [batch, num_preds, hidden] - Tensor containing the hidden representations
            of predicted spans or relations.
        gold_reps (Tensor): [batch, num_golds, hidden] - Tensor containing the hidden representations
            of gold (true) spans or relations.
        neg_limit (float): A large negative value used for masking elements during the calculation
            to ensure they do not affect the maximum similarity computation.
        pred_labels (Tensor, optional): [batch, num_preds] or [batch, num_preds, num_classes] -
            Labels for each predicted entry. If multilabel, should be binary vectors; if unilabel, single integers.
        gold_labels (Tensor, optional): [batch, num_golds] or [batch, num_golds, num_classes] -
            Labels for each gold entry. Shape and type should match `pred_labels`.
        pred_masks (Tensor, optional): [batch, num_preds] - Binary mask indicating valid predicted entries.
        gold_masks (Tensor, optional): [batch, num_golds] - Binary mask indicating valid gold entries.

    Returns:
        Tensor: Scalar tensor representing the cosine similarity loss. This loss is computed as 1 minus
            the maximum cosine similarity between matched entries, averaged or summed over all valid gold entries,
            adjusted for any specified masking.
    """
    # Normalize embeddings for cosine similarity calculation
    pred_norm = F.normalize(pred_reps, dim=-1)
    gold_norm = F.normalize(gold_reps, dim=-1)
    # Compute cosine similarities: [batch, num_golds, num_preds]
    cosine_scores = torch.einsum('bgh,bph->bgp', gold_norm, pred_norm)

    #apply label matching if labels are provided otherwise ignores labels (binary labels)
    if pred_labels is not None and gold_labels is not None:
    # Determine if labels are multilabel or unilabel and apply appropriate matching logic
        if gold_labels.dim() == 3:
            # Multilabel scenario: binary vectors
            label_match = (gold_labels.unsqueeze(2) & pred_labels.unsqueeze(1)).sum(dim=-1) > 0
        else:
            # Unilabel scenario: single integers
            label_match = gold_labels.unsqueeze(2) == pred_labels.unsqueeze(1)

        # Set unmatchable entries to a large negative value to exclude them from max
        cosine_scores = cosine_scores.masked_fill(~label_match, neg_limit)

    # Apply prediction masks if provided
    if pred_masks is not None:
        cosine_scores = cosine_scores.masked_fill(~pred_masks.unsqueeze(1), neg_limit)

    # Get max similarity for each gold (i.e., best matching pred with the same label)
    max_sim_gold, _ = cosine_scores.max(dim=-1)  # [batch, num_golds]
    #default to -1 if no matches
    max_sim_gold[max_sim_gold == neg_limit] = 0

    loss = 1 - max_sim_gold

    # Apply gold mask if provided
    if gold_masks is not None:
        valid_count = gold_masks.sum()
        if valid_count == 0:
            return torch.tensor(0.0, device=loss.device, requires_grad=True)
        loss = loss * gold_masks
        return loss.sum() / valid_count
    else:
        return loss.mean()