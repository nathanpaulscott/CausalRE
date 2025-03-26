'''
TO DO:
This is future work, you can do this later when you have the model up and running.....

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
from abc import ABC, abstractmethod

from .layers_other import ProjectionLayer
#for testing
#from layers_other import ProjectionLayer
#from utils import set_all_seeds



############################################################################
#Helper functions###########################################################
############################################################################
############################################################################

############################################################################
#NEW CODE#################################### for the pruned rels scenario
############################################################################
def map_rel_ids_to_span_ids(rel_ids, span_filter_map):
    """
    Maps rel_ids (which reference span_ids) to span_ids using span_filter_map.

    Args:
        rel_ids (torch.Tensor): Shape (batch, num_rel, 2), indices of relations in span_ids.
        span_filter_map (torch.Tensor): Shape (batch, num_spans), maps span_ids to span_ids.

    Returns:
        torch.Tensor: Shape (batch, num_rel, 2), mapped indices for candidate span representations.
    """
    batch_size, num_rel, _ = rel_ids.shape

    #Extract head and tail span indices from rel_ids
    head_span_indices = rel_ids[:, :, 0]  # (batch, num_rel)
    tail_span_indices = rel_ids[:, :, 1]  # (batch, num_rel)
    #Map head and tail indices to candidate span IDs
    head_span_ids = torch.gather(span_filter_map, dim=1, index=head_span_indices)
    tail_span_ids = torch.gather(span_filter_map, dim=1, index=tail_span_indices)
    #Stack back into rel_ids format (batch, num_rel, 2)
    mapped_rel_ids = torch.stack([head_span_ids, tail_span_ids], dim=-1)

    return mapped_rel_ids  # (batch, num_rel, 2)


def make_head_tail_rel_reps_from_pruned_ids(span_reps, mapped_rel_ids):
    """
    Constructs relation representations from pruned and mapped rel_ids.

    Args:
        span_reps (torch.Tensor): Shape (batch, num_spans, hidden), span representations.
        mapped_rel_ids (torch.Tensor): Shape (batch, num_rel, 2), indices of valid relation pairs (head, tail).

    Returns:
        torch.Tensor: Shape (batch, num_rel, hidden), relation representations.
    """
    batch_size, num_rel, _ = mapped_rel_ids.shape
    hidden_size = span_reps.shape[-1]
    # Extract head and tail representations using mapped rel_ids
    head_reps = torch.gather(span_reps, dim=1, index=mapped_rel_ids[:, :, 0].unsqueeze(-1).expand(-1, -1, hidden_size))
    tail_reps = torch.gather(span_reps, dim=1, index=mapped_rel_ids[:, :, 1].unsqueeze(-1).expand(-1, -1, hidden_size))
    # Concatenate and project relation representations
    rel_reps = torch.cat([head_reps, tail_reps], dim=-1)  # (batch, num_rel, hidden_size * 2)
    
    return rel_reps  # (batch, num_rel, hidden_size * 2)


def make_context_rel_reps_from_pruned_ids(mapped_rel_ids, token_reps, token_masks, neg_limit, context_mask_fn, context_reps_fn, **kwargs):
    """
    Constructs relation representations for pruned rel_ids while incorporating context representations.

    Args:
        span_reps (torch.Tensor): Shape (batch, num_spans, hidden), span representations.
        rel_ids (torch.Tensor): Shape (batch, num_rel, 2), pruned relation indices.
        span_filter_map (torch.Tensor): Shape (batch, num_spans), mapping span_ids to span_ids.
        token_reps (torch.Tensor): Shape (batch, seq_len, hidden), contextual token representations.
        token_masks (torch.Tensor): Shape (batch, seq_len), token-level masks.
        neg_limit: Value representing invalid positions in token representations.
        context_func: Function to generate context masks (specific to each context strategy).
        **kwargs: Additional parameters for context function.

    Returns:
        torch.Tensor: Shape (batch, num_rel, hidden), relation representations with context.
    """
    #Generate context representations **only** for valid relations
    context_masks = context_mask_fn(token_masks, mapped_rel_ids, **kwargs)  # Get context mask for these relations
    context_reps = context_reps_fn(token_reps, context_masks, neg_limit)  # Compute context representation
    return context_reps  # (batch, num_rel, hidden_size)

############################################################################
############################################################################
############################################################################





def make_head_tail_reps_old(span_reps, top_k_spans):
    """
    Expands candidate span representations to generate head and tail representations for all span pairs.
    Args:
        span_reps (torch.Tensor): A tensor of shape (batch, top_k_spans, hidden) containing span representations.
        top_k_spans (int): The number of candidate spans per batch.
    Returns:
        tuple: (head_reps, tail_reps) with shape (batch, top_k_spans, top_k_spans, hidden).
    """
    head_reps = span_reps.unsqueeze(2).expand(-1, top_k_spans, top_k_spans, -1)  # (batch, top_k_spans, top_k_spans, hidden)
    tail_reps = span_reps.unsqueeze(1).expand(-1, top_k_spans, top_k_spans, -1)  # (batch, top_k_spans, top_k_spans, hidden)
    return head_reps, tail_reps



def make_head_tail_reps(span_reps, rel_ids):
    """
    Generates head and tail span representations for relation classification.

    Args:
        span_reps (torch.Tensor): A tensor of shape (batch, num_spans, hidden) containing span representations.
        rel_ids (torch.Tensor): A tensor of shape (batch, num_relations, 2), where each entry is (head_id, tail_id).
    
    Returns:
        tuple:
            - head_reps (torch.Tensor): A tensor of shape (batch, num_relations, hidden), representing the head spans.
            - tail_reps (torch.Tensor): A tensor of shape (batch, num_relations, hidden), representing the tail spans.
    """
    _, _, hidden = span_reps.shape

    # Extract valid head & tail span representations using rel_ids
    head_ids, tail_ids = rel_ids[..., 0], rel_ids[..., 1]  # Shape: (batch, num_relations)
    # Gather head and tail span representations using rel_ids
    head_reps = torch.gather(span_reps, 1, head_ids.unsqueeze(-1).expand(-1, -1, hidden))
    tail_reps = torch.gather(span_reps, 1, tail_ids.unsqueeze(-1).expand(-1, -1, hidden))

    return head_reps, tail_reps



def make_seq_indices_expanded(token_masks):
    """
    Creates an expanded tensor of sequence indices that matches the shape required for relation mask computations.
    Args:
        token_masks (torch.Tensor): A tensor of shape (batch, seq_len)
    Returns:
        torch.Tensor: A tensor of shape (batch, 1, 1, seq_len) containing sequence indices, suitable for broadcast operations.
    """
    #makes a tensor of seq indices in the shape of rel_masks
    seq_indices = torch.arange(token_masks.shape[1], device=token_masks.device)   #(seq_len)
    seq_indices = seq_indices.unsqueeze(0).expand(token_masks.shape[0], -1).unsqueeze(1).unsqueeze(1)   #(batch, 1, 1, seq_len)
    return seq_indices


def get_span_start_end_ids(span_ids):
    """
    Extracts the start and end token indices for each candidate span and prepares them for broadcast operations.
    Args:
        span_ids (torch.Tensor): A tensor of shape (batch, top_k_spans, 2) where the last dimension contains 
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
    head_start_ids = span_ids[:, :, 0].unsqueeze(2)  # (batch, top_k_spans, 1)
    head_end_ids = span_ids[:, :, 1].unsqueeze(2)    # (batch, top_k_spans, 1)
    tail_start_ids = span_ids[:, :, 0].unsqueeze(1)  # (batch, 1, top_k_spans)
    tail_end_ids = span_ids[:, :, 1].unsqueeze(1)    # (batch, 1, top_k_spans)
    return head_start_ids, head_end_ids, tail_start_ids, tail_end_ids


def make_span_masks(seq_inds, start_ids, end_ids):
    """
    Constructs the span token mask, for the given span
    it sets the mask to 1 for tokens in the given span
    NOTE: The end_ids are the actual end + 1 (Python style), so the logic accounts for this.
    Args:
        seq_inds (torch.Tensor): Tensor of sequence indices with shape (batch, 1, 1, seq_len).
        start_ids (torch.Tensor): Tensor of span start indices (batch, top_k_spans, 1).
        end_ids (torch.Tensor): Tensor of span end indices (batch, top_k_spans, 1).
    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - window_masks: A boolean mask of shape (batch, top_k_spans, seq_len) indicating tokens within the window.
            - span_masks: A boolean mask of shape (batch, top_k_spans, seq_len) indicating tokens inside the span.
    """
    #add a last dim so it is compatible for broadcasting with seq_inds shape
    start_ids = start_ids.unsqueeze(-1)
    end_ids = end_ids.unsqueeze(-1)
    #make the span mask, will be 1 for tokens between start and end-1
    return (seq_inds >= start_ids) & (seq_inds <= end_ids - 1)


############################################################################
#NO CONTEXT ALGO############################################################
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
    def __init__(self, hidden_size, ffn_ratio, dropout, layer_type, **kwargs):
        super().__init__()
        self.out_layer = ProjectionLayer(input_dim  = hidden_size * 2,
                                         ffn_ratio  = ffn_ratio, 
                                         output_dim = hidden_size, 
                                         dropout    = dropout,
                                         layer_type = layer_type)
        #no weight init required as is handled by the out_layer


    def forward_old(self, span_reps, **kwargs):
        """
        Generates relation representations from candidate span representations by forming all possible 
        head-tail combinations and reprojecting them into a hidden space.
        Args:
            span_reps (torch.Tensor): A tensor of shape (batch, top_k_spans, hidden) containing 
                span representations for candidate spans.
            **kwargs: Additional keyword arguments (unused).
        Returns:
            torch.Tensor: A tensor of shape (batch, top_k_spans**2, hidden), where each entry represents 
                a relation representation for a pair of spans.
        """
        #this works on the full catesian product of span_ids x span_ids
        batch, top_k_spans, _ = span_reps.shape
        #Expand heads n tails
        head_reps, tail_reps = make_head_tail_reps_old(span_reps, top_k_spans)
        #make the rel reps
        rel_reps = torch.cat([head_reps, tail_reps], dim=-1)   #(batch, top_k_spans, top_k_spans, hidden * 2)
        #move the shape back to 3 dims
        rel_reps = rel_reps.view(batch, top_k_spans * top_k_spans, -1)   #(batch, top_k_spans**2, hidden*2)
        #Apply output layer and return
        return self.out_layer(rel_reps)    #(batch, top_k_spans**2, hidden)


    def forward(self, span_reps, rel_ids, **kwargs):
        """
        Generates relation representations from candidate span representations by forming all possible 
        head-tail combinations and reprojecting them into a hidden space.
        Args:
            span_reps (torch.Tensor): A tensor of shape (batch, top_k_spans, hidden) containing 
                span representations for candidate spans.
            **kwargs: Additional keyword arguments (unused).
        Returns:
            torch.Tensor: A tensor of shape (batch, top_k_spans**2, hidden), where each entry represents 
                a relation representation for a pair of spans.
        """
        #this works on the full catesian product of span_ids x span_ids
        batch, num_spans, _ = span_reps.shape
        #get head and tail reps
        head_reps, tail_reps = make_head_tail_reps(span_reps, rel_ids)    #(b, num_rels, hidden), (b, num_rels, hidden)
        #make the rel reps
        rel_reps = torch.cat([head_reps, tail_reps], dim=-1)   #(batch, num_rels, hidden * 2)
        #Apply output layer and return
        return self.out_layer(rel_reps)    #(batch, num_rels, hidden)


############################################################################
#CONTEXT ALGOS##############################################################
############################################################################

#use a base class for context cases as they are all very similar except for the differences in the context mask calcs

class RelRepContextBase(nn.Module, ABC):
    """
    Abstract base for context-based relation representations combining head, tail, and context.
    """
    def __init__(self, hidden_size, ffn_ratio, dropout, no_context_rep, context_pooling, layer_type, **kwargs):
        super().__init__()
        self.hidden_size = hidden_size

        self.no_context_rep = no_context_rep
        if no_context_rep == 'emb':
            self.no_context_embedding = nn.Parameter(torch.randn(hidden_size) * 0.01)

        self.context_pooling = context_pooling

        self.out_layer = ProjectionLayer(input_dim  = hidden_size * 3,
                                         ffn_ratio  = ffn_ratio, 
                                         output_dim = hidden_size, 
                                         dropout    = dropout,
                                         layer_type = layer_type)


    @abstractmethod
    def make_base_context_masks(self, seq_indices, head_start_ids, head_end_ids, tail_start_ids, tail_end_ids):
        """
        Subclasses should implement this method to return base context masks
        This is the algorithm specific function, eg. window context, between context
        """
        pass


    def make_context_masks(self, token_masks, span_ids, **kwargs):
        '''
        This makes the context masks, the base_context_masks are the algo specific part
        '''
        rel_masks = kwargs.get("rel_masks")
        if rel_masks is None:
            raise ValueError("rel_masks must be provided as a keyword argument.")
        #get the common params, make the seq_indices and start/end ids
        batch, top_k_spans, _ = span_ids.shape
        seq_indices = make_seq_indices_expanded(token_masks)
        head_start_ids, head_end_ids, tail_start_ids, tail_end_ids = get_span_start_end_ids(span_ids)
        #make the exclusion masks
        head_span_masks = make_span_masks(seq_indices, head_start_ids, head_end_ids)
        tail_span_masks = make_span_masks(seq_indices, tail_start_ids, tail_end_ids)
        exclusion_masks = ~(head_span_masks | tail_span_masks)
        #expand token/rel masks
        expanded_rel_masks = rel_masks.view(batch, top_k_spans, top_k_spans).unsqueeze(-1)
        expanded_token_masks = token_masks.unsqueeze(1).unsqueeze(1)
 
        ###################################################
        #make the context specific base_context_masks
        base_context_masks = self.make_base_context_masks(seq_indices, head_start_ids, head_end_ids, tail_start_ids, tail_end_ids)
        ###################################################
 
        # Apply all masks
        return base_context_masks & exclusion_masks & expanded_rel_masks & expanded_token_masks


    def no_context_handler(self, context_reps, neg_limit):
        """
        Applies a fallback mechanism for cases where the pooled context representation is entirely invalid.
        For each relation representation in `context_reps` that is entirely equal to `neg_limit` across the hidden dimension,
        this function replaces that representation with a fallback value. The fallback is determined by the `no_context_rep` parameter:
        - If 'zero', the invalid context is replaced with a zero vector.
        - If 'emb', the invalid context is replaced with a learned no-context embedding (provided via `no_context_embedding`).
        Args:
            context_reps (torch.Tensor): A tensor of shape (batch, top_k_spans, top_k_spans, hidden) representing pooled context representations.
            neg_limit: A scalar value used to mark invalid token representations prior to pooling.
        """
        all_masked = (context_reps == neg_limit).all(dim=-1)
        if all_masked.any():
            if self.no_context_rep == 'zero':
                context_reps[all_masked] = 0

            elif self.no_context_rep == 'emb':
                context_reps[all_masked] = self.no_context_embedding.type_as(context_reps).expand_as(context_reps[all_masked])

            else:
                raise Exception('rel_no_context_rep value is invalid')


    def make_context_reps(self, token_reps, context_masks, neg_limit):
        """
        Computes the context representation by applying the context mask to token representations.
        Invalid token positions are replaced with neg_limit, then a max pooling is performed over the sequence dimension.
        A fallback is applied for cases where the pooled representation is entirely invalid.
        Args:
            token_reps (torch.Tensor): Tensor of shape (batch, seq_len, hidden).
            context_masks (torch.Tensor): Boolean tensor of shape (batch, top_k_spans, top_k_spans, seq_len).
            neg_limit: Scalar value used to mark invalid token positions prior to max pooling.
        Returns:
            torch.Tensor: Tensor of shape (batch, top_k_spans, top_k_spans, hidden) representing the pooled context.
        """
        #make the context reps by expanding hte token reps, filling the masked out token reps with the neg_limit and max pooling on the seq dim
        context_reps, _ = token_reps.unsqueeze(1).unsqueeze(1).masked_fill(~context_masks.unsqueeze(-1), neg_limit).max(dim=3)
        #Apply fallback for entirely invalid context representations and return
        #NOTE: it modifies context_reps, no return from this function
        self.no_context_handler(context_reps, neg_limit)
        return context_reps


    def forward(self, span_reps, span_ids, token_reps, token_masks, neg_limit, **kwargs):
        """
        Computes relation representations by concatenating the head/tail and algo specific context reps
        """
        batch, top_k_spans, _ = span_reps.shape  # (batch, top_k_spans, hidden)
        #Span representation expansion for relational context.
        head_reps, tail_reps = make_head_tail_reps_old(span_reps, top_k_spans)
        #Generate the context mask (subclass-specific)
        context_masks = self.make_context_masks(token_masks, span_ids, **kwargs)
        context_reps = self.make_context_reps(token_reps, context_masks, neg_limit)
        #Concatenate head, tail, and combined context representations.
        rel_reps = torch.cat([head_reps, tail_reps, context_reps], dim=-1)  # (batch, top_k_spans, top_k_spans, 3 * hidden)
        #Apply output layer and reshape.
        rel_reps = self.out_layer(rel_reps)     # (batch, top_k_spans, top_k_spans, hidden)
        return rel_reps.view(batch, top_k_spans * top_k_spans, -1)  # (batch, top_k_spans**2, hidden)


############################################################################
#BETWEEN CONTEXT ALGO#######################################################
############################################################################

class RelRepBetweenContext(RelRepContextBase):
    def make_base_context_masks(self, seq_indices, head_start_ids, head_end_ids, tail_start_ids, tail_end_ids):
        """
        Generates masks for tokens between head and tail spans.
        As is the the base context, it has not yet excluded the tokens in the spans
        """
        #Compute the between masks
        min_start = torch.min(head_start_ids, tail_start_ids)
        max_end = torch.max(head_end_ids, tail_end_ids)
        return make_span_masks(seq_indices, min_start, max_end)



############################################################################
#WINDOW CONTEXT ALGO########################################################
############################################################################

class RelRepWindowContext(RelRepContextBase):
    def __init__(self, hidden_size, ffn_ratio, dropout, no_context_rep, context_pooling, layer_type, window_size, **kwargs):
        super().__init__(hidden_size, ffn_ratio, dropout, no_context_rep, context_pooling, layer_type, **kwargs)
        self.window_size = window_size

    def make_window_masks(self, seq_inds, start_ids, end_ids):
        """
        Constructs window masks for a given set of start and end indices.
        The window masks select tokens within a window before the span start or after the span end.
        NOTE: The end_ids are the actual end + 1 (Python style), so the logic accounts for this.
        Args:
            seq_inds (torch.Tensor): Tensor of sequence indices with shape (batch, 1, 1, seq_len).
            start_ids (torch.Tensor): Tensor of span start indices (batch, top_k_spans, 1).
            end_ids (torch.Tensor): Tensor of span end indices (batch, top_k_spans, 1).
        Returns:
           - window_masks: A boolean mask of shape (batch, top_k_spans, seq_len) indicating tokens within the window.
        """
        win = self.window_size
        #add a last dim so it is compatible for broadcasting with seq_inds shape
        start_ids = start_ids.unsqueeze(-1)
        end_ids = end_ids.unsqueeze(-1)
        #make the masks
        start_masks = (seq_inds >= (start_ids - win)) & (seq_inds < start_ids)
        end_masks = (seq_inds > end_ids - 1) & (seq_inds <= (end_ids - 1 + win))   #due to the end id being actual + 1 (python style)
        #logical OR them for the window masks
        return start_masks | end_masks
        

    def make_base_context_masks(self, seq_indices, head_start_ids, head_end_ids, tail_start_ids, tail_end_ids):
        """
        Generates masks for tokens in a window around the head and tail spans.
        As is the the base context, it has not yet excluded the tokens in the spans
        """
        #make the window masks
        head_window_masks = self.make_window_masks(seq_indices, head_start_ids, head_end_ids)
        tail_window_masks = self.make_window_masks(seq_indices, tail_start_ids, tail_end_ids)
        return head_window_masks | tail_window_masks



############################################################################
#BETWEEN and WINDOW CONTEXT ALGO############################################
############################################################################


class RelRepBetweenWindowContext(RelRepContextBase):
    def __init__(self, hidden_size, ffn_ratio, dropout, no_context_rep, context_pooling, layer_type, window_size, **kwargs):
        super().__init__(hidden_size, ffn_ratio, dropout, no_context_rep, context_pooling, layer_type, **kwargs)
        self.window_size = window_size


    def make_window_masks(self, seq_inds, start_ids, end_ids):
        """
        Constructs window masks for a given set of start and end indices.
        The window masks select tokens within a window before the span start or after the span end.
        NOTE: The end_ids are the actual end + 1 (Python style), so the logic accounts for this.
        Args:
            seq_inds (torch.Tensor): Tensor of sequence indices with shape (batch, 1, 1, seq_len).
            start_ids (torch.Tensor): Tensor of span start indices (batch, top_k_spans, 1).
            end_ids (torch.Tensor): Tensor of span end indices (batch, top_k_spans, 1).
        Returns:
           - window_masks: A boolean mask of shape (batch, top_k_spans, seq_len) indicating tokens within the window.
        """
        win = self.window_size
        #add a last dim so it is compatible for broadcasting with seq_inds shape
        start_ids = start_ids.unsqueeze(-1)
        end_ids = end_ids.unsqueeze(-1)
        #make the masks
        start_masks = (seq_inds >= (start_ids - win)) & (seq_inds < start_ids)
        end_masks = (seq_inds > end_ids - 1) & (seq_inds <= (end_ids - 1 + win))   #due to the end id being actual + 1 (python style)
        #logical OR them for the window masks
        return start_masks | end_masks
        

    def make_base_context_masks(self, seq_indices, head_start_ids, head_end_ids, tail_start_ids, tail_end_ids):
        """
        Generates masks for tokens in a window around the head and tail spans.
        As is the the base context, it has not yet excluded the tokens in the spans
        """
        #Compute the between masks
        min_start = torch.min(head_start_ids, tail_start_ids)
        max_end = torch.max(head_end_ids, tail_end_ids)
        between_masks = make_span_masks(seq_indices, min_start, max_end)
        #make the window masks
        head_window_masks = self.make_window_masks(seq_indices, head_start_ids, head_end_ids)
        tail_window_masks = self.make_window_masks(seq_indices, tail_start_ids, tail_end_ids)
        return head_window_masks | tail_window_masks | between_masks




############################################################################
#Mother Class###############################################################
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
            dropout     = config.dropout,       # Dropout rate for the FFN
            layer_type  = config.output_layer_type    #whether to use simple or ffn layer for the output projection
            ...                              # Additional parameters specific to the chosen strategy
            eg. 
            ffn_ratio   = config.ffn_ratio,     # Expansion ratio for the FFN
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
        elif rel_mode == 'between_window_context':
            self.rel_rep_layer = RelRepBetweenWindowContext(**kwargs)
        else:
            raise ValueError(f'Unknown rel mode {rel_mode}')



    def forward(self, **kwargs):
        """
        Computes relation representations using the selected submodule.
        
        The forward pass expects keyword arguments that are appropriate for the chosen strategy. For example, if using 
        a context-based strategy, the arguments may include:
        
            - span_reps: Tensor of shape (batch, top_k_spans, hidden) containing span representations.
            - span_ids: Tensor of shape (batch, top_k_spans, 2) containing span indices (start and end positions).
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







##############################
# Test code
##############################
def test_no_context():
    print("=== Test No-Context ===")
    batch, top_k_spans, hidden = 2, 3, 4
    span_reps = torch.randn(batch, top_k_spans, hidden)
    model = RelRepNoContext(hidden_size=hidden, ffn_ratio=1.0, dropout=0.0)
    out = model(span_reps=span_reps)
    print("Output shape:", out.shape)  # Expected (2, 9, 4)
    #print(out)

def test_between_context():
    print("\n=== Test Between-Context ===")
    batch, top_k_spans, hidden, seq_len = 2, 3, 4, 5
    span_reps = torch.randn(batch, top_k_spans, hidden)
    # Create dummy candidate span indices; here each span is defined as [start, end]
    span_ids = torch.tensor([[[0, 1], [1, 2], [2, 3]],
                                  [[0, 1], [2, 3], [4, 5]]])
    token_reps = torch.randn(batch, seq_len, hidden)
    token_masks = torch.ones(batch, seq_len, dtype=torch.bool)
    rel_masks = ~torch.eye(top_k_spans, dtype=torch.bool).unsqueeze(0).expand(batch, -1, -1)
    neg_limit = -6.5e4
    model = RelRepBetweenContext(hidden_size=hidden, ffn_ratio=1.0, dropout=0.0,
                                 no_context_rep='emb', context_pooling='max')
    out = model(span_reps=span_reps, span_ids=span_ids,
                token_reps=token_reps, token_masks=token_masks, neg_limit=neg_limit,
                rel_masks=rel_masks)
    print("Output shape:", out.shape)  # Expected (2, 9, 4)
    print(out)
    #print(rel_masks)
    #print(token_reps)

def test_window_context():
    print("\n=== Test Window-Context ===")
    batch, top_k_spans, hidden, seq_len = 2, 3, 4, 5
    span_reps = torch.randn(batch, top_k_spans, hidden)
    span_ids = torch.tensor([[[0, 1], [1, 2], [2, 3]],
                                  [[0, 1], [2, 3], [3, 4]]])
    token_reps = torch.randn(batch, seq_len, hidden)
    token_masks = torch.ones(batch, seq_len, dtype=torch.bool)
    rel_masks = ~torch.eye(top_k_spans, dtype=torch.bool).unsqueeze(0).expand(batch, -1, -1)
    neg_limit = -6.5e4
    model = RelRepWindowContext(hidden_size=hidden, ffn_ratio=1.0, dropout=0.0,
                                window_size=2, no_context_rep='zero', context_pooling='max')
    out = model(span_reps=span_reps, span_ids=span_ids,
                token_reps=token_reps, token_masks=token_masks, neg_limit=neg_limit,
                rel_masks=rel_masks)
    print("Output shape:", out.shape)  # Expected (2, 9, 4)
    print(out)
    #print(token_reps)

def test_relation_rep_layer():
    print("\n=== Test RelationRepLayer (Mother Class) ===")
    batch, top_k_spans, hidden, seq_len = 2, 3, 4, 5
    span_reps = torch.randn(batch, top_k_spans, hidden)
    span_ids = torch.tensor([[[0, 1], [1, 2], [2, 3]],
                                  [[0, 1], [2, 3], [2, 4]]])
    token_reps = torch.randn(batch, seq_len, hidden)
    token_masks = torch.ones(batch, seq_len, dtype=torch.bool)
    rel_masks = ~torch.eye(top_k_spans, dtype=torch.bool).unsqueeze(0).expand(batch, -1, -1)
    neg_limit = -6.5e4

    # No-context test
    print("\n-> Testing no_context mode")
    model = RelationRepLayer(rel_mode='no_context', hidden_size=hidden, ffn_ratio=1.0, dropout=0.0)
    out = model(span_reps=span_reps)
    print("Output shape:", out.shape)

    # Between-context test
    print("\n-> Testing between_context mode")
    model = RelationRepLayer(rel_mode='between_context', hidden_size=hidden, ffn_ratio=1.0,
                             dropout=0.0, no_context_rep='zero', context_pooling='max')
    out = model(span_reps=span_reps, span_ids=span_ids,
                token_reps=token_reps, token_masks=token_masks, neg_limit=neg_limit,
                rel_masks=rel_masks)
    print("Output shape:", out.shape)

    # Window-context test
    print("\n-> Testing window_context mode")
    model = RelationRepLayer(rel_mode='window_context', hidden_size=hidden, ffn_ratio=1.0,
                             dropout=0.0, window_size=2, no_context_rep='zero', context_pooling='max')
    out = model(span_reps=span_reps, span_ids=span_ids,
                token_reps=token_reps, token_masks=token_masks, neg_limit=neg_limit,
                rel_masks=rel_masks)
    print("Output shape:", out.shape)

if __name__ == "__main__":
    set_all_seeds(seed=42)
    #test_no_context()
    test_between_context()
    #test_window_context()
    #test_relation_rep_layer()





