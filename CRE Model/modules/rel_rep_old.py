'''
TO DO:
This is future work, you can do this later when you have the model up and running.....

- add in the simple reprojection layer option over the FFN projection, it is more simple, FFN may not be required, need to ablate it
- I would add one to combine window with between, i.e. just have the outer window + the beetween
- have not exhaustively tested the window verison
- add some kind of attention pooling over maxpooling
    Some ideas for attention pooling:
    - the slow way would be to get the context tokens for each pair (batch, top_k, top+k, seq_len, hidden) => HUGE tensor, then attention pool them with the query being the concat of the head and tail span reps, then reproject to hidden or just concat the size of 2*hidden to the head/tail spans
    NOTE: this method can be used for any of the rel_rep variants
    - the faster way (for the window case only) => attention pool the context tokens for each span, i.e. get the window tokens, use the span rep as the query.  Then make the concatenated combination of the context reps for each pair, then either reproject down to hidden (maybe) or just concat both to the head and tail span reps, then reproject all down to hidden
    NOTE: can not do above for the between case as between tokens are dependent on knowing both spans in the pair.
    - think of some other attention pooling scenarios to try
'''


import torch
from torch import nn
import torch.nn.init as init

#from .layers_other import FFNProjectionLayer
#for testing
from layers_other import FFNProjectionLayer
from utils import set_all_seeds


############################################################################
#Helper functions###########################################################
############################################################################
############################################################################


def make_head_tail_reps(cand_span_reps, top_k_spans):
    """
    Expands candidate span representations to generate head and tail representations for all span pairs.
    
    Args:
        cand_span_reps (torch.Tensor): A tensor of shape (batch, top_k_spans, hidden) containing span representations.
        top_k_spans (int): The number of candidate spans per batch.
    
    Returns:
        tuple: A tuple (head_reps, tail_reps) where:
            - head_reps (torch.Tensor): Tensor of shape (batch, top_k_spans, top_k_spans, hidden) representing the head 
              representations for each span pair.
            - tail_reps (torch.Tensor): Tensor of shape (batch, top_k_spans, top_k_spans, hidden) representing the tail 
              representations for each span pair.
    """
    head_reps = cand_span_reps.unsqueeze(2).expand(-1, top_k_spans, top_k_spans, -1)  # (batch, top_k_spans, top_k_spans, hidden)
    tail_reps = cand_span_reps.unsqueeze(1).expand(-1, top_k_spans, top_k_spans, -1)  # (batch, top_k_spans, top_k_spans, hidden)
    return head_reps, tail_reps


def make_seq_indices_expanded(token_reps):
    """
    Creates an expanded tensor of sequence indices that matches the shape required for relation mask computations.
    
    Args:
        token_reps (torch.Tensor): A tensor of shape (batch, seq_len, hidden) representing token-level features.
    
    Returns:
        torch.Tensor: A tensor of shape (batch, 1, 1, seq_len) containing sequence indices, suitable for broadcast operations.
    """
    #makes a tensor of seq indices in the shape of rel_masks
    seq_indices = torch.arange(token_reps.shape[1], device=token_reps.device)   #(seq_len)
    seq_indices = seq_indices.unsqueeze(0).expand(token_reps.shape[0], -1).unsqueeze(1).unsqueeze(1)   #(batch, 1, 1, seq_len)
    return seq_indices


def get_span_start_end_ids(cand_span_ids):
    """
    Extracts the start and end token indices for each candidate span and prepares them for broadcast operations.
    
    Args:
        cand_span_ids (torch.Tensor): A tensor of shape (batch, top_k_spans, 2) where the last dimension contains 
            the start and end indices of each span.
    
    Returns:
        tuple: A tuple containing:
            - head_start_ids (torch.Tensor): Tensor of shape (batch, top_k_spans, 1) with the span start indices.
            - head_end_ids (torch.Tensor): Tensor of shape (batch, top_k_spans, 1) with the span end indices.
            - tail_start_ids (torch.Tensor): Tensor of shape (batch, 1, top_k_spans) with the span start indices 
              (for pairing).
            - tail_end_ids (torch.Tensor): Tensor of shape (batch, 1, top_k_spans) with the span end indices 
              (for pairing).
    """
    head_start_ids = cand_span_ids[:, :, 0].unsqueeze(2)  # (batch, top_k_spans, 1)
    head_end_ids = cand_span_ids[:, :, 1].unsqueeze(2)    # (batch, top_k_spans, 1)
    tail_start_ids = cand_span_ids[:, :, 0].unsqueeze(1)  # (batch, 1, top_k_spans)
    tail_end_ids = cand_span_ids[:, :, 1].unsqueeze(1)    # (batch, 1, top_k_spans)
    return head_start_ids, head_end_ids, tail_start_ids, tail_end_ids




def no_context_handler(context_reps, neg_limit, no_context_rep, no_context_embedding=None):
    """
    Applies a fallback mechanism for cases where the pooled context representation is entirely invalid.
    
    For each relation representation in `context_reps` that is entirely equal to `neg_limit` across the hidden dimension,
    this function replaces that representation with a fallback value. The fallback is determined by the `no_context_rep` parameter:
      - If 'zero', the invalid context is replaced with a zero vector.
      - If 'emb', the invalid context is replaced with a learned no-context embedding (provided via `no_context_embedding`).
    
    Args:
        context_reps (torch.Tensor): A tensor of shape (..., hidden) representing pooled context representations.
        neg_limit: A scalar value used to mark invalid token representations prior to pooling.
        no_context_rep (str): Strategy for handling missing context. Must be either 'zero' or 'emb'.
        no_context_embedding (torch.Tensor, optional): A learned embedding vector of shape (hidden,) used as fallback when 
            no_context_rep is 'emb'. Defaults to None.
    
    Raises:
        Exception: If `no_context_rep` is not 'zero' or 'emb'.
    """
    all_masked = (context_reps == neg_limit).all(dim=-1)
    if all_masked.any():
        if no_context_rep == 'zero':
            context_reps[all_masked] = 0
        elif no_context_rep == 'emb':
            expanded_no_context_embedding = no_context_embedding.unsqueeze(0).unsqueeze(0).unsqueeze(0)
            expanded_no_context_embedding = expanded_no_context_embedding.expand_as(context_reps[all_masked])
            context_reps[all_masked] = expanded_no_context_embedding
        else:
            raise Exception('rel_no_context_rep value is invalid')


############################################################################
############################################################################
############################################################################
############################################################################



class RelRepNoContext(nn.Module):
    """
    Constructs relation representations by concatenating head and tail span representations 
    and projecting the result into a hidden space. This variant does not incorporate any 
    additional contextual information beyond the span representations themselves.
    
    Attributes:
        out_layer (nn.Module): A feedforward projection layer that takes concatenated head and tail 
            representations (of dimensionality 2 * hidden_size) and projects them into a hidden 
            space of size hidden_size.
    
    Args:
        hidden_size (int): Dimensionality of the input span representations.
        ffn_ratio (float): Expansion ratio for the intermediate layer in the feedforward network.
        dropout (float): Dropout rate applied within the feedforward network.
        **kwargs: Additional keyword arguments (unused in this class).
    """
    def __init__(self, hidden_size, ffn_ratio, dropout, **kwargs):
        super().__init__()
        self.out_layer = FFNProjectionLayer(input_dim  = hidden_size * 2,
                                            ffn_ratio  = ffn_ratio, 
                                            output_dim = hidden_size, 
                                            dropout    = dropout)
        
        #no weight init required as is handled by the out_layer

    def forward(self, cand_span_reps, **kwargs):
        """
        Generates relation representations from candidate span representations by forming all possible 
        head-tail combinations and reprojecting them into a hidden space.
        
        Args:
            cand_span_reps (torch.Tensor): A tensor of shape (batch, top_k_spans, hidden) containing 
                span representations for candidate spans.
            **kwargs: Additional keyword arguments (unused).
        
        Returns:
            torch.Tensor: A tensor of shape (batch, top_k_spans**2, hidden), where each entry represents 
                a relation representation for a pair of spans.
        """
        batch, top_k_spans, _ = cand_span_reps.shape
        
        #Expand heads n tails
        head_reps, tail_reps = make_head_tail_reps(cand_span_reps, top_k_spans)
        #make the rel reps
        rel_reps = torch.cat([head_reps, tail_reps], dim=-1)   #(batch, top_k_spans, top_k_spans, hidden * 2)
        #Apply output layer
        rel_reps = self.out_layer(rel_reps)    #(batch, top_k_spans, top_k_spans, hidden)
        #move the shape back to 3 dims
        #rel_reps = rel_reps.view(batch, top_k_spans * top_k_spans, -1)   #(batch, top_k_spans**2, hidden)

        return rel_reps    #(batch, top_k_spans**2, hidden)


######################################################################
######################################################################
######################################################################
######################################################################
######################################################################



class RelRepBetweenContext(nn.Module):
    """
    Constructs relation representations by concatenating head and tail span representations 
    with a pooled "between" context representation derived from the tokens lying between the spans.
    The concatenated representation is then projected into a hidden space using an FFNProjectionLayer.
    
    The between context is computed as follows:
      - The tokens between the head and tail spans are identified based on their start and end indices.
      - A boolean mask is generated to select tokens that lie between the spans and are valid according to 
        the provided token mask and relation mask.
      - If valid tokens exist, a max pooling operation is applied over these tokens to produce a fixed-dimensional 
        context representation.
      - If no valid tokens exist (i.e. the pooled representation equals the negative limit for all elements), 
        the context is replaced with a fallback representation. This fallback is determined by the `no_context_rep` 
        parameter: either a vector of zeros (if set to `'zero'`) or a learned no-context embedding (if set to `'emb'`).
    
    Attributes:
        out_layer (nn.Module): A feedforward projection layer that takes the concatenated head, tail, and 
            context representations (with total dimension 3 * hidden_size) and projects them into a hidden space 
            of dimension hidden_size.
    
    Args:
        hidden_size (int): Dimensionality of the span and token representations.
        ffn_ratio (float): Expansion ratio for the intermediate layer in the feedforward network.
        dropout (float): Dropout rate used within the feedforward network.
        no_context_rep (str): Specifies the fallback for when no valid context tokens are found.
            Accepted values are 'zero' (to use a zero vector) or 'emb' (to use a learned no-context embedding).
        context_pooling: Specifies the pooling strategy for context tokens (e.g., max pooling). This parameter 
            is included for potential future extensions and currently, max pooling is applied.
        **kwargs: Additional keyword arguments (unused in this class).
    """
    def __init__(self, hidden_size, ffn_ratio, dropout, no_context_rep, context_pooling, **kwargs):
        super().__init__()
        self.no_context_rep = no_context_rep
        if no_context_rep == 'emb':
            self.no_context_embedding = nn.Parameter(torch.randn(hidden_size) * 0.01)  #Small random initialization
        self.context_pooling = context_pooling
        
        self.out_layer = FFNProjectionLayer(input_dim = hidden_size * 3,
                                            ffn_ratio = ffn_ratio, 
                                            output_dim = hidden_size, 
                                            dropout = dropout)
        #no weight init required as is handled by the FFNLayer


    def make_context_masks(self, token_reps, token_masks, rel_masks, cand_span_ids, batch, top_k_spans):
        """
        Constructs the context mask for the between-context case. This mask identifies the tokens that lie 
        between the head and tail spans and are valid according to the token and relation masks.
        
        Args:
            token_reps (torch.Tensor): Tensor of shape (batch, seq_len, hidden) containing token representations.
            token_masks (torch.Tensor): Tensor of shape (batch, seq_len) where True indicates valid tokens.
            rel_masks (torch.Tensor): Boolean tensor reshaped to (batch, top_k_spans, top_k_spans) that masks out invalid relations.
            cand_span_ids (torch.Tensor): Tensor of shape (batch, top_k_spans, 2) containing start and end indices for spans.
            batch (int): Batch size.
            top_k_spans (int): Number of candidate spans per batch obs.
        
        Returns:
            torch.Tensor: A boolean mask of shape (batch, top_k_spans, top_k_spans, seq_len) indicating which tokens 
            are considered as valid "between" context.
        """
        head_start_ids, head_end_ids, tail_start_ids, tail_end_ids = get_span_start_end_ids(cand_span_ids)
        #Compute "between" bounds
        min_end = torch.min(head_end_ids, tail_end_ids)       # (batch, top_k_spans, top_k_spans)
        max_start = torch.max(head_start_ids, tail_start_ids) # (batch, top_k_spans, top_k_spans)
        #Mask to identify valid "between" tokens
        #which is when the min end occurs before the max start, otherwise the spans overlap and there are no between tokens
        #it also ands the mask with the rel_mask, to mask out invalid relations
        #thus valid_between will be a mask indicating which relations have 'between' context tokens
        valid_between = (min_end < max_start) & rel_masks.view(batch, top_k_spans, top_k_spans)   # (batch, top_k_spans, top_k_spans)
        #now we create the actual between token context mask, i.e. the mask to select only the between tokens in each relation sequence
        #NOTE: this will use a lot of memory as these tensors are huge, but they are boolean and do not have the hidden dim yet
        #make the seq indices in expanded format
        seq_indices = make_seq_indices_expanded(token_reps)   #(batch, 1, 1, seq_len)
        expanded_min_end = min_end.unsqueeze(-1)  # (batch, top_k_spans, top_k_spans, 1)
        expanded_max_start = max_start.unsqueeze(-1)  # (batch, top_k_spans, top_k_spans, 1)
        expanded_valid_between = valid_between.unsqueeze(-1)  # (batch, top_k_spans, top_k_spans, 1)
        #Get boolean masks for tokens in valid "between" ranges and mixin the token masks
        context_masks = (seq_indices >= expanded_min_end) & \
                        (seq_indices < expanded_max_start) & \
                        expanded_valid_between & \
                        token_masks.unsqueeze(1).unsqueeze(1) #(batch, top_k_spans, top_k_spans, seq_len)
        return context_masks


    def make_context_reps(self, token_reps, context_masks, neg_limit):
        """
        Generates context representations by applying the provided context mask on token representations.
        Invalid tokens (as determined by the mask) are set to neg_limit, and a max pooling is performed over 
        the sequence dimension.
        
        If the resulting context representation is entirely equal to neg_limit (indicating no valid context tokens),
        a fallback is applied: either a zero vector or a learned no-context embedding is used, depending on the 
        `no_context_rep` setting.
        
        Args:
            token_reps (torch.Tensor): Tensor of shape (batch, seq_len, hidden) containing token representations.
            context_masks (torch.Tensor): Boolean mask of shape (batch, top_k_spans, top_k_spans, seq_len) for valid context tokens.
            neg_limit: A scalar value used to mark invalid token representations prior to pooling.
        
        Returns:
            torch.Tensor: A tensor of shape (batch, top_k_spans, top_k_spans, hidden) representing the pooled context.
        """
        #this expands the token reps to (batch, top_k_spans, top_k_spans, seq_len, hidden)
        #then applies the rel_specific context mask to set masked out tokens to neg_limit
        #then max pools over the seq dim (dim 3)
        context_reps, _ = token_reps.unsqueeze(1).unsqueeze(1).masked_fill(~context_masks.unsqueeze(-1), neg_limit).max(dim=3)  # (batch, top_k_spans, top_k_spans, hidden)
        #edge case handler where all the context reps are set to neg_limit => no valid context
        #in this case replace with a learned no context embedding
        no_context_handler(context_reps, neg_limit, self.no_context_rep, 
                           None if self.no_context_rep == 'zero' else self.no_context_embedding)
        return context_reps


    def forward(self, cand_span_reps, cand_span_ids, token_reps, token_masks, rel_masks, neg_limit, **kwargs):
        """
        Computes relation representations by combining head, tail, and between-context representations.
        
        Args:
            cand_span_reps (torch.Tensor): Tensor of shape (batch, top_k_spans, hidden) containing span representations.
            cand_span_ids (torch.Tensor): Tensor of shape (batch, top_k_spans, 2) with start and end indices for each span.
            token_reps (torch.Tensor): Tensor of shape (batch, seq_len, hidden) containing token-level representations.
            token_masks (torch.Tensor): Tensor of shape (batch, seq_len) where True indicates valid tokens.
            rel_masks (torch.Tensor): Boolean tensor (reshaped as needed) masking out invalid relations.
            neg_limit: A scalar used to mark invalid tokens prior to pooling.
            **kwargs: Additional keyword arguments (unused).
        
        Returns:
            torch.Tensor: A tensor of shape (batch, top_k_spans**2, hidden) where each entry represents a relation 
            representation for a pair of spans, composed of concatenated head, tail, and context representations.
        """
        batch, top_k_spans, hidden = cand_span_reps.shape

        #Make the head and tail reps
        head_reps, tail_reps = make_head_tail_reps(cand_span_reps, top_k_spans)

        #Make the context reps
        context_masks = self.make_context_masks(token_reps, token_masks, rel_masks, cand_span_ids, batch, top_k_spans)
        context_reps = self.make_context_reps(token_reps, context_masks, neg_limit)

        # Concatenate head, tail, and context representations
        rel_reps = torch.cat([head_reps, tail_reps, context_reps], dim=-1)  # (batch, top_k_spans, top_k_spans, hidden*3)

        # Apply output layer and reshape
        rel_reps = self.out_layer(rel_reps)     # (batch, top_k_spans, top_k_spans, hidden)
        rel_reps = rel_reps.view(batch, top_k_spans * top_k_spans, -1)  # (batch, top_k_spans**2, hidden)

        return rel_reps


############################################################################
############################################################################
############################################################################
############################################################################




class RelRepWindowContext(nn.Module):
    """
    Constructs relation representations by combining head, tail, and window-based context representations.
    
    This module computes relation representations for span pairs by:
      1. Expanding candidate span representations into head and tail representations.
      2. Extracting a context representation from tokens surrounding each span. Tokens are selected 
         from a window (of size `window_size`) immediately before and after the span boundaries. 
         Tokens that fall within the span are excluded.
      3. Invalid tokens (as indicated by the token mask) are replaced with a negative limit (neg_limit) 
         prior to pooling. A max pooling operation is then applied over the sequence dimension to yield a 
         fixed-dimensional context representation.
      4. If no valid context tokens are found (i.e. the pooled representation equals neg_limit in every 
         hidden dimension), a fallback is applied. The fallback is determined by the `no_context_rep` parameter:
         either a zero vector is used (if set to `'zero'`) or a learned no-context embedding is applied (if set 
         to `'emb'`).
      5. Finally, the concatenated head, tail, and context representations (total dimension 3 * hidden_size) 
         are projected into a hidden space via a feedforward layer, and the output is reshaped to a tensor of 
         shape (batch, top_k_spans**2, hidden).
    
    Attributes:
        out_layer (nn.Module): A feedforward projection layer (FFNProjectionLayer) that projects the concatenated 
            head, tail, and context representations into a hidden space of dimension hidden_size.
        window_size (int): The number of tokens to consider before and after each span for context pooling.
    
    Args:
        hidden_size (int): Dimensionality of the input span and token representations.
        ffn_ratio (float): Expansion ratio for the intermediate layer in the feedforward network.
        dropout (float): Dropout rate applied within the feedforward network.
        window_size (int): Size of the window (in tokens) for context extraction around the spans.
        no_context_rep (str): Strategy for handling cases with no valid context tokens. Accepted values are:
            'zero' - to use a zero vector, or 'emb' - to use a learned no-context embedding.
        context_pooling: Specifies the pooling strategy for context tokens (currently, max pooling is used).
        **kwargs: Additional keyword arguments (unused in this class).
    """
    def __init__(self, hidden_size, ffn_ratio, dropout, window_size, no_context_rep, context_pooling, **kwargs):
        super().__init__()
        self.window_size = window_size
        self.no_context_rep = no_context_rep
        if no_context_rep == 'emb':
            self.no_context_embedding = nn.Parameter(torch.randn(hidden_size) * 0.01)  #Small random initialization
        self.context_pooling = context_pooling
        
        self.out_layer = FFNProjectionLayer(input_dim = hidden_size * 3,
                                            ffn_ratio = ffn_ratio, 
                                            output_dim = hidden_size, 
                                            dropout = dropout)
        

    def make_window_masks(self, seq_inds, start_ids, end_ids):
        """
        Constructs window masks for a given set of start and end indices.
        
        The mask selects tokens that fall within a window before the span start or after the span end.
        
        Args:
            seq_inds (torch.Tensor): Tensor of sequence indices with shape (batch, 1, 1, seq_len).
            start_ids (torch.Tensor): Tensor of span start indices (broadcastable to match seq_inds).
            end_ids (torch.Tensor): Tensor of span end indices (broadcastable to match seq_inds).
        
        Returns:
            torch.Tensor: A boolean mask of shape (batch, top_k_spans, top_k_spans, seq_len) indicating 
            tokens within the specified window.
        """
        win = self.window_size
        start_masks = ((seq_inds >= (start_ids - win)) & (seq_inds < start_ids))  #(batch, top_k_spans, top_k_spans, seq_len)
        end_masks =   ((seq_inds > end_ids) & (seq_inds <= (end_ids + win)))      #(batch, top_k_spans, top_k_spans, seq_len)
        return start_masks | end_masks                                            #(batch, top_k_spans, top_k_spans, seq_len)



    def make_context_masks(self, token_reps, token_masks, cand_span_ids):
        """
        Generates a combined context mask for window-based context extraction.
        
        The mask identifies tokens that lie within the window before or after each span,
        excluding tokens inside the span and incorporating the token validity mask.
        
        Args:
            token_reps (torch.Tensor): Tensor of shape (batch, seq_len, hidden) containing token representations.
            token_masks (torch.Tensor): Boolean tensor of shape (batch, seq_len) indicating valid tokens.
            cand_span_ids (torch.Tensor): Tensor of shape (batch, top_k_spans, 2) with start and end indices for spans.
        
        Returns:
            torch.Tensor: A boolean mask of shape (batch, top_k_spans, top_k_spans, seq_len) for selecting valid window context tokens.
        """
        #make the seq indices in expanded format
        seq_indices = make_seq_indices_expanded(token_reps)   #(batch, 1, 1, seq_len)
        #Get start/end ids
        head_start_ids, head_end_ids, tail_start_ids, tail_end_ids = get_span_start_end_ids(cand_span_ids)
        #make the head and tail window masks
        head_window_mask = self.make_window_masks(seq_indices, head_start_ids, head_end_ids)
        tail_window_mask = self.make_window_masks(seq_indices, tail_start_ids, tail_end_ids)
        #Combine the head and tail masks to get a single mask for the context and mix in the token masks
        combined_context_mask = (head_window_mask | tail_window_mask) & \
                                token_masks.unsqueeze(1).unsqueeze(1)    # (batch, top_k_spans, top_k_spans, seq_len)
        return combined_context_mask


    def make_context_reps(self, token_reps, context_masks, neg_limit):
        """
        Computes the context representation by applying the window context mask to token representations.
        
        Invalid token positions are set to neg_limit prior to pooling, and a max pooling operation is applied
        over the sequence dimension. If the pooled representation consists entirely of neg_limit values (indicating 
        no valid context tokens), a fallback is applied as determined by `no_context_rep` (either a zero vector or 
        a learned no-context embedding).
        
        Args:
            token_reps (torch.Tensor): Tensor of shape (batch, seq_len, hidden) containing token representations.
            context_masks (torch.Tensor): Boolean mask of shape (batch, top_k_spans, top_k_spans, seq_len) indicating valid tokens.
            neg_limit: A scalar value used to mark invalid token positions before pooling.
        
        Returns:
            torch.Tensor: A tensor of shape (batch, top_k_spans, top_k_spans, hidden) representing the pooled window context.
        """
        #this expands the token reps to (batch, top_k_spans, top_k_spans, seq_len, hidden)
        #then applies the rel_specific context mask to set masked out tokens to neg_limit
        #then max pools over the seq dim (dim 3)
        context_reps, _ = token_reps.unsqueeze(1).unsqueeze(1).masked_fill(~context_masks.unsqueeze(-1), neg_limit).max(dim=3)  # (batch, top_k_spans, top_k_spans, hidden)
        #edge case handler where all the context reps are set to neg_limit => no valid context
        #in this case replace with a learned no context embedding
        no_context_handler(context_reps, neg_limit, self.no_context_rep, 
                           None if self.no_context_rep == 'zero' else self.no_context_embedding)

        return context_reps


    def forward(self, cand_span_reps, cand_span_ids, token_reps, token_masks, neg_limit, **kwargs):
        """
        Computes relation representations using window-based context extraction.
        
        The process involves:
          1. Expanding candidate span representations to generate head and tail representations.
          2. Generating a window-based context mask from token representations using the candidate span indices.
          3. Pooling token representations (with invalid tokens replaced by neg_limit) within the window to form 
             the context representation.
          4. Applying a fallback for cases with no valid context tokens (using either a zero vector or a learned 
             no-context embedding).
          5. Concatenating the head, tail, and context representations.
          6. Projecting the concatenated representation into a hidden space and reshaping the output.
        
        Args:
            cand_span_reps (torch.Tensor): Tensor of shape (batch, top_k_spans, hidden) containing span representations.
            cand_span_ids (torch.Tensor): Tensor of shape (batch, top_k_spans, 2) with start and end indices for spans.
            token_reps (torch.Tensor): Tensor of shape (batch, seq_len, hidden) containing token-level representations.
            token_masks (torch.Tensor): Boolean tensor of shape (batch, seq_len) indicating valid tokens.
            neg_limit: A scalar value used to mark invalid token positions prior to pooling.
            **kwargs: Additional keyword arguments (unused).
        
        Returns:
            torch.Tensor: A tensor of shape (batch, top_k_spans**2, hidden) where each entry represents a relation 
            representation for a pair of spans, composed of concatenated head, tail, and window context representations.
        """
        batch, top_k_spans, hidden = cand_span_reps.shape  # (batch, top_k_spans, hidden)

        # Span representation expansion for relational context
        head_reps, tail_reps = make_head_tail_reps(cand_span_reps, top_k_spans)
        
        #Context reps
        ####################################################
        context_masks = self.make_context_masks(token_reps, token_masks, cand_span_ids)
        context_reps = self.make_context_reps(token_reps, context_masks, neg_limit)

        # Concatenate head, tail, and combined context representations
        rel_reps = torch.cat([head_reps, tail_reps, context_reps], dim=-1)  # (batch, top_k_spans, top_k_spans, 3 * hidden)

        # Apply output layer and reshape
        rel_reps = self.out_layer(rel_reps)     # (batch, top_k_spans, top_k_spans, hidden)
        rel_reps = rel_reps.view(batch, top_k_spans * top_k_spans, -1)  # (batch, top_k_spans**2, hidden)

        return rel_reps        




############################################################################
#Mother Class###############################################################
############################################################################
############################################################################


class RelationRepLayer(nn.Module):
    """
    Constructs relation representations by selecting one of several strategies for combining span representations.
    
    This module dynamically instantiates one of the following submodules based on the `rel_mode` parameter:
    
      - **no_context**: Uses only head and tail span representations (by concatenation) to form relation representations.
      
      - **between_context**: Combines head and tail span representations with a pooled context representation derived 
        from tokens between the spans. (Note: This mode applies a between-context mechanism that replaces invalid context 
        with a fallback, either a zero vector or a learned embedding.)
      
      - **window_context**: Combines head and tail span representations with a pooled context representation computed 
        from tokens around the spans within a specified window. (Note: This mode extracts tokens from a window before and 
        after each span, excludes tokens inside the span, and applies pooling.)
    
    The chosen strategy is determined by the `rel_mode` argument, and any additional keyword arguments (`**kwargs`) are 
    forwarded to the respective submodule.
    
    **Example Initialization:**
    
        self.rel_rep_layer = RelationRepLayer(
            rel_mode    = config.rel_mode,    # 'no_context', 'between_context', or 'window_context'
            hidden_size = config.hidden_size,   # Dimensionality for span representations
            ffn_ratio   = config.ffn_ratio,     # Expansion ratio for the FFN
            dropout     = config.dropout,       # Dropout rate for the FFN
            ...                              # Additional parameters specific to the chosen strategy
        )
    
    Args:
        rel_mode (str): The relation representation strategy to use. Must be one of:
            - 'no_context'
            - 'between_context'
            - 'window_context'
        **kwargs: Additional keyword arguments passed to the instantiated relation representation submodule.
    
    """
    def __init__(self, rel_mode, **kwargs):
        super().__init__()
        self.rel_mode = rel_mode
    
        if rel_mode == 'no_context':
            self.rel_rep_layer = RelRepNoContext(**kwargs)
        elif rel_mode == 'between_context':
            self.rel_rep_layer = RelRepBetweenContext(**kwargs)
        elif rel_mode == 'window_context':
            self.rel_rep_layer = RelRepWindowContext(**kwargs)
        else:
            raise ValueError(f'Unknown rel mode {rel_mode}')



    def forward(self, **kwargs):
        """
        Computes relation representations using the selected submodule.
        
        The forward pass expects keyword arguments that are appropriate for the chosen strategy. For example, if using 
        a context-based strategy, the arguments may include:
        
            - cand_span_reps: Tensor of shape (batch, top_k_spans, hidden) containing span representations.
            - cand_span_ids: Tensor of shape (batch, top_k_spans, 2) containing span indices (start and end positions).
            - token_reps: Tensor of shape (batch, seq_len, hidden) containing token-level representations.
            - token_masks: Tensor of shape (batch, seq_len) indicating valid tokens.
            - rel_masks: Boolean tensor masking invalid relations.
            - neg_limit: A scalar marking invalid token positions before pooling.
        
        Returns:
            torch.Tensor: A tensor representing the relation representations. Typically, the output shape is 
            (batch, top_k_spans**2, hidden), though this may vary depending on the chosen strategy.
        """
        result = self.rel_rep_layer(**kwargs)

        return result






#Trying to get this test code working
#Trying to get this test code working
#Trying to get this test code working
#Trying to get this test code working
#Trying to get this test code working
#Trying to get this test code working
#Trying to get this test code working

##############################
# Test code
##############################
def test_no_context():
    print("=== Test No-Context ===")
    batch, top_k_spans, hidden = 2, 3, 4
    cand_span_reps = torch.randn(batch, top_k_spans, hidden)
    model = RelRepNoContext(hidden_size=hidden, ffn_ratio=1.0, dropout=0.0)
    out = model(cand_span_reps=cand_span_reps)
    print("Output shape:", out.shape)  # Expected (2, 9, 4)
    print(out)

def test_between_context():
    print("\n=== Test Between-Context ===")
    batch, top_k_spans, hidden, seq_len = 2, 3, 4, 5
    cand_span_reps = torch.randn(batch, top_k_spans, hidden)
    # Create dummy candidate span indices; here each span is defined as [start, end]
    cand_span_ids = torch.tensor([[[0, 1], [1, 2], [2, 3]],
                                  [[0, 1], [2, 3], [3, 4]]])
    token_reps = torch.randn(batch, seq_len, hidden)
    token_masks = torch.ones(batch, seq_len, dtype=torch.bool)
    rel_masks = ~torch.eye(top_k_spans, dtype=torch.bool).unsqueeze(0).expand(batch, -1, -1)
    neg_limit = -6.5e4
    model = RelRepBetweenContext(hidden_size=hidden, ffn_ratio=1.0, dropout=0.0,
                                 no_context_rep='zero', context_pooling='max')
    out = model(cand_span_reps=cand_span_reps, cand_span_ids=cand_span_ids,
                token_reps=token_reps, token_masks=token_masks, neg_limit=neg_limit,
                rel_masks=rel_masks)
    print("Output shape:", out.shape)  # Expected (2, 9, 4)
    print(out)

def test_window_context():
    print("\n=== Test Window-Context ===")
    batch, top_k_spans, hidden, seq_len = 2, 3, 4, 5
    cand_span_reps = torch.randn(batch, top_k_spans, hidden)
    cand_span_ids = torch.tensor([[[0, 1], [1, 2], [2, 3]],
                                  [[0, 1], [2, 3], [3, 4]]])
    token_reps = torch.randn(batch, seq_len, hidden)
    token_masks = torch.ones(batch, seq_len, dtype=torch.bool)
    rel_masks = ~torch.eye(top_k_spans, dtype=torch.bool).unsqueeze(0).expand(batch, -1, -1)
    neg_limit = -6.5e4
    model = RelRepWindowContext(hidden_size=hidden, ffn_ratio=1.0, dropout=0.0,
                                window_size=1, no_context_rep='zero', context_pooling='max')
    out = model(cand_span_reps=cand_span_reps, cand_span_ids=cand_span_ids,
                token_reps=token_reps, token_masks=token_masks, neg_limit=neg_limit)
    print("Output shape:", out.shape)  # Expected (2, 9, 4)
    print(out)

def test_relation_rep_layer():
    print("\n=== Test RelationRepLayer (Mother Class) ===")
    batch, top_k_spans, hidden, seq_len = 2, 3, 4, 5
    cand_span_reps = torch.randn(batch, top_k_spans, hidden)
    cand_span_ids = torch.tensor([[[0, 1], [1, 2], [2, 3]],
                                  [[0, 1], [2, 3], [2, 4]]])
    token_reps = torch.randn(batch, seq_len, hidden)
    token_masks = torch.ones(batch, seq_len, dtype=torch.bool)
    rel_masks = torch.ones(batch, top_k_spans, top_k_spans, dtype=torch.bool)
    neg_limit = -6.5e4

    # No-context test
    print("\n-> Testing no_context mode")
    model = RelationRepLayer(rel_mode='no_context', hidden_size=hidden, ffn_ratio=1.0, dropout=0.0)
    out = model(cand_span_reps=cand_span_reps)
    print("Output shape:", out.shape)

    # Between-context test
    print("\n-> Testing between_context mode")
    model = RelationRepLayer(rel_mode='between_context', hidden_size=hidden, ffn_ratio=1.0,
                             dropout=0.0, no_context_rep='zero', context_pooling='max')
    out = model(cand_span_reps=cand_span_reps, cand_span_ids=cand_span_ids,
                token_reps=token_reps, token_masks=token_masks, neg_limit=neg_limit,
                rel_masks=rel_masks)
    print("Output shape:", out.shape)

    # Window-context test
    print("\n-> Testing window_context mode")
    model = RelationRepLayer(rel_mode='window_context', hidden_size=hidden, ffn_ratio=1.0,
                             dropout=0.0, window_size=1, no_context_rep='zero', context_pooling='max')
    out = model(cand_span_reps=cand_span_reps, cand_span_ids=cand_span_ids,
                token_reps=token_reps, token_masks=token_masks, neg_limit=neg_limit)
    print("Output shape:", out.shape)

if __name__ == "__main__":
    set_all_seeds(seed=42)
    #test_no_context()
    test_between_context()
    #test_window_context()
    #test_relation_rep_layer()


