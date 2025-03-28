import torch
from torch import nn
import torch.nn.init as init
from abc import ABC, abstractmethod

from .layers_other import ProjectionLayer
#for testing
#from layers_other import ProjectionLayer
#from utils import set_all_seeds




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



    def make_head_tail_reps_for_rel_ids(self, span_reps, rel_ids):
        """
        Extracts head and tail span representations for given relation indices.
        NOTE: the ids in rel_ids are aligned to span_ids/reps

        Args:
            span_reps (torch.Tensor): A tensor of shape (batch, num_spans, hidden) containing span representations.
            rel_ids (torch.Tensor): A tensor of shape (batch, num_relations, 2), where each entry is (head_id, tail_id) from span_ids.
        
        Returns:
            tuple:
                - head_reps (torch.Tensor): A tensor of shape (batch, num_relations, hidden), representing the head spans.
                - tail_reps (torch.Tensor): A tensor of shape (batch, num_relations, hidden), representing the tail spans.
        """
        _, _, hidden = span_reps.shape
        # Extract valid head & tail span representations using rel_ids
        head_ids = rel_ids[..., 0]  # Shape: (batch, num_relations)
        tail_ids = rel_ids[..., 1]  # Shape: (batch, num_relations)
        # Gather head and tail span representations using rel_ids
        head_reps = torch.gather(span_reps, 1, head_ids.unsqueeze(-1).expand(-1, -1, hidden))
        tail_reps = torch.gather(span_reps, 1, tail_ids.unsqueeze(-1).expand(-1, -1, hidden))

        return head_reps, tail_reps    #(batch, num_relations, hidden), (batch, num_relations, hidden)



    def forward(self, span_reps, rel_ids, **kwargs):
        """
        Generates relation representations from candidate span representations by forming all possible 
        head-tail combinations and reprojecting them into a hidden space.
        Args:
            span_reps (torch.Tensor): A tensor of shape (batch, num_spans, hidden) containing 
                span representations for candidate spans.
            **kwargs: Additional keyword arguments (unused).
        Returns:
            torch.Tensor: A tensor of shape (batch, num_spans**2, hidden), where each entry represents 
                a relation representation for a pair of spans.
        """
        #get head and tail reps
        head_reps, tail_reps = self.make_head_tail_reps_for_rel_ids(span_reps, rel_ids)    #(b, num_rels, hidden), (b, num_rels, hidden)
        #concat
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


    def make_head_tail_reps_for_rel_ids(self, span_reps, rel_ids):
        """
        Extracts head and tail span representations for given relation indices.
        NOTE: the ids in rel_ids are aligned to span_ids

        Args:
            span_reps (torch.Tensor): A tensor of shape (batch, num_spans, hidden) containing span representations.
            rel_ids (torch.Tensor): A tensor of shape (batch, num_relations, 2), where each entry is (head_id, tail_id) from span_ids.
        
        Returns:
            tuple:
                - head_reps (torch.Tensor): A tensor of shape (batch, num_relations, hidden), representing the head spans.
                - tail_reps (torch.Tensor): A tensor of shape (batch, num_relations, hidden), representing the tail spans.
        """
        _, _, hidden = span_reps.shape
        # Extract valid head & tail span representations using rel_ids
        head_ids = rel_ids[..., 0]  # Shape: (batch, num_relations)
        tail_ids = rel_ids[..., 1]  # Shape: (batch, num_relations)
        # Gather head and tail span representations using rel_ids
        head_reps = torch.gather(span_reps, 1, head_ids.unsqueeze(-1).expand(-1, -1, hidden))
        tail_reps = torch.gather(span_reps, 1, tail_ids.unsqueeze(-1).expand(-1, -1, hidden))

        return head_reps, tail_reps    #(batch, num_relations, hidden), (batch, num_relations, hidden)


    @abstractmethod
    def make_base_context_masks(self, seq_indices, head_start_ids, head_end_ids, tail_start_ids, tail_end_ids):
        """
        Subclasses should implement this method to return base context masks
        This is the algorithm specific function, eg. window context, between context
        """
        pass



    def make_context_masks(self, token_masks, span_ids, rel_ids, rel_masks):
        '''
        makes the context masks which indicate which token reps to include as context        
        returns masks of shape (b, num_rels, seq_len) bool
        '''
        _, seq_len = token_masks.shape
        device = token_masks.device

        #make the exclusion masks, where the head/tail spans exist
        # Extract head and tail indices from span_ids
        head_start = torch.gather(span_ids[:, :, 0], 1, rel_ids[:, :, 0])
        head_end = torch.gather(span_ids[:, :, 1], 1, rel_ids[:, :, 0])
        tail_start = torch.gather(span_ids[:, :, 0], 1, rel_ids[:, :, 1])
        tail_end = torch.gather(span_ids[:, :, 1], 1, rel_ids[:, :, 1])
        # Create masks for tokens in head and tail spans
        seq_indices = torch.arange(seq_len, device=device).unsqueeze(0).unsqueeze(0)
        head_masks = (seq_indices >= head_start.unsqueeze(-1)) & (seq_indices < head_end.unsqueeze(-1))
        tail_masks = (seq_indices >= tail_start.unsqueeze(-1)) & (seq_indices < tail_end.unsqueeze(-1))
        # Combine masks and expand to all tokens in spans
        exclusion_masks = ~(head_masks | tail_masks)    #(b, num_rels, seq_len)

        #expand token and rel masks to broadcastable dims
        expanded_rel_masks = rel_masks.unsqueeze(-1)      #(b, num_rels, 1)
        expanded_token_masks = token_masks.unsqueeze(1)   #(b, 1, seq_len)

        ###################################################
        #make the context specific base_context_masks
        base_context_masks = self.make_base_context_masks(seq_indices, head_start, head_end, tail_start, tail_end)
        ###################################################
 
        # Apply all masks
        return base_context_masks & exclusion_masks & expanded_rel_masks & expanded_token_masks   #(b, num_rels, seq_len) bool


    def no_context_handler(self, context_reps, neg_limit):
        """
        Applies a fallback mechanism for cases where the pooled context representation is entirely invalid.
        For each relation representation in `context_reps` that is entirely equal to `neg_limit` across the hidden dimension,
        this function replaces that representation with a fallback value. The fallback is determined by the `no_context_rep` parameter:
        - If 'zero', the invalid context is replaced with a zero vector.
        - If 'emb', the invalid context is replaced with a learned no-context embedding (provided via `no_context_embedding`).
        Args:
            context_reps (torch.Tensor): A tensor of shape (batch, num_spans, num_spans, hidden) representing pooled context representations.
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


    def make_context_reps(self, token_reps, cls_reps, rel_ids, neg_limit, context_masks):
        batch, num_rels, _ = rel_ids.shape
        _, seq_len, hidden = token_reps.shape

        if cls_reps is not None:
            return cls_reps.unsqueeze(1).expand(-1, num_rels, -1)

        # Calculate context representations
        context_reps, _ = token_reps.unsqueeze(1).masked_fill(~context_masks.unsqueeze(-1), neg_limit).max(dim=2)
        # Handle no-context cases
        self.no_context_handler(context_reps, neg_limit)

        return context_reps


    def forward(self, token_reps, token_masks, span_reps, span_ids, rel_ids, rel_masks, neg_limit, cls_reps=None, **kwargs):
        """
        Computes relation representations by concatenating the head/tail and algo specific context reps
        """
        #get head and tail reps
        head_reps, tail_reps = self.make_head_tail_reps_for_rel_ids(span_reps, rel_ids)    #(b, num_rels, hidden), (b, num_rels, hidden)
        #Generate the context mask (subclass-specific)
        context_masks = self.make_context_masks(token_masks, span_ids, rel_ids, rel_masks)
        context_reps = self.make_context_reps(token_reps, cls_reps, rel_ids, neg_limit, context_masks)
        #concat everything
        rel_reps = torch.cat([head_reps, tail_reps, context_reps], dim=-1)  # (batch, num_rels, 3 * hidden)
        #Apply output layer and reshape.
        return self.out_layer(rel_reps)     # (batch, num_rels, hidden)


############################################################################
#BETWEEN CONTEXT ALGO#######################################################
############################################################################

class RelRepBetweenContext(RelRepContextBase):
    def make_base_context_masks(self, seq_indices, head_start, head_end, tail_start, tail_end):
        """
        Generates masks for tokens between head and tail spans.
        As is the the base context, it has not yet excluded the tokens in the spans
        """
        min_start = torch.min(head_start, tail_start).unsqueeze(-1)
        max_end = torch.max(head_end, tail_end).unsqueeze(-1)
        #make the span mask, will be 1 for tokens between start and end-1
        return (seq_indices >= min_start) & (seq_indices < max_end)



############################################################################
#WINDOW CONTEXT ALGO########################################################
############################################################################

class RelRepWindowContext(RelRepContextBase):
    def __init__(self, hidden_size, ffn_ratio, dropout, no_context_rep, context_pooling, layer_type, window_size, **kwargs):
        super().__init__(hidden_size, ffn_ratio, dropout, no_context_rep, context_pooling, layer_type, **kwargs)
        self.window_size = window_size

    def make_window_masks(self, seq_indices, start_ids, end_ids):
        """
        Constructs window masks for a given set of start and end indices.
        The window masks select tokens within a window before the span start or after the span end.
        NOTE: The end_ids are the actual end + 1 (Python style), so the logic accounts for this.
        Args:
            seq_indices (torch.Tensor): Tensor of sequence indices with shape (batch, 1, 1, seq_len).
            start_ids (torch.Tensor): Tensor of span start indices (batch, num_spans, 1).
            end_ids (torch.Tensor): Tensor of span end indices (batch, num_spans, 1).
        Returns:
           - window_masks: A boolean mask of shape (batch, num_spans, seq_len) indicating tokens within the window.
        """
        win = self.window_size
        #add a last dim so it is compatible for broadcasting with seq_indices shape
        start_ids = start_ids.unsqueeze(-1)
        end_ids = end_ids.unsqueeze(-1)
        #make the masks
        start_masks = (seq_indices >= (start_ids - win)) & (seq_indices < start_ids)
        end_masks = (seq_indices > end_ids - 1) & (seq_indices <= (end_ids - 1 + win))   #due to the end id being actual + 1 (python style)
        #logical OR them for the window masks
        return start_masks | end_masks
        

    def make_base_context_masks(self, seq_indices, head_start, head_end, tail_start, tail_end):
        """
        Generates masks for tokens in a window around the head and tail spans.
        As is the the base context, it has not yet excluded the tokens in the spans
        """
        #make the window masks
        head_window_masks = self.make_window_masks(seq_indices, head_start, head_end)
        tail_window_masks = self.make_window_masks(seq_indices, tail_start, tail_end)
        return head_window_masks | tail_window_masks







class RelRepBetweenWindowContext(RelRepContextBase):
    def __init__(self, hidden_size, ffn_ratio, dropout, no_context_rep, context_pooling, layer_type, window_size, **kwargs):
        super().__init__(hidden_size, ffn_ratio, dropout, no_context_rep, context_pooling, layer_type, **kwargs)
        self.window_size = window_size

    def make_window_masks(self, seq_indices, start_ids, end_ids):
        """
        Constructs window masks for a given set of start and end indices.
        The window masks select tokens within a window before the span start or after the span end.
        NOTE: The end_ids are the actual end + 1 (Python style), so the logic accounts for this.
        Args:
            seq_indices (torch.Tensor): Tensor of sequence indices with shape (batch, 1, 1, seq_len).
            start_ids (torch.Tensor): Tensor of span start indices (batch, num_spans, 1).
            end_ids (torch.Tensor): Tensor of span end indices (batch, num_spans, 1).
        Returns:
           - window_masks: A boolean mask of shape (batch, num_spans, seq_len) indicating tokens within the window.
        """
        win = self.window_size
        #add a last dim so it is compatible for broadcasting with seq_indices shape
        start_ids = start_ids.unsqueeze(-1)
        end_ids = end_ids.unsqueeze(-1)
        #make the masks
        start_masks = (seq_indices >= (start_ids - win)) & (seq_indices < start_ids)
        end_masks = (seq_indices > end_ids - 1) & (seq_indices <= (end_ids - 1 + win))   #due to the end id being actual + 1 (python style)
        #logical OR them for the window masks
        return start_masks | end_masks
        

    def make_base_context_masks(self, seq_indices, head_start, head_end, tail_start, tail_end):
        """
        Generates masks for tokens in a window around the head and tail spans and the tokens between the spans.
        As is the the base context, it has not yet excluded the tokens in the spans
        """
        #make the window masks
        head_window_masks = self.make_window_masks(seq_indices, head_start, head_end)
        tail_window_masks = self.make_window_masks(seq_indices, tail_start, tail_end)

        #make the between masks
        min_start = torch.min(head_start, tail_start).unsqueeze(-1)
        max_end = torch.max(head_end, tail_end).unsqueeze(-1)
        #make the span mask, will be 1 for tokens between start and end-1
        between_masks = (seq_indices >= min_start) & (seq_indices < max_end)

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
        
            - span_reps: Tensor of shape (batch, num_spans, hidden) containing span representations.
            - span_ids: Tensor of shape (batch, num_spans, 2) containing span indices (start and end positions).
            - token_reps: Tensor of shape (batch, seq_len, hidden) containing token-level representations.
            - token_masks: Tensor of shape (batch, seq_len) indicating valid tokens.
            - rel_masks: Boolean tensor masking invalid relations.
            - neg_limit: A scalar marking invalid token positions before pooling.
        
        Returns:
            torch.Tensor: A tensor representing the relation representations. Typically, the output shape is 
            (batch, num_spans**2, hidden), though this may vary depending on the chosen strategy.
        """
        result = self.rel_rep_layer(**kwargs)

        return result







##############################
# Test code
##############################
def test_no_context():
    print("=== Test No-Context ===")
    batch, num_spans, hidden = 2, 3, 4
    span_reps = torch.randn(batch, num_spans, hidden)
    model = RelRepNoContext(hidden_size=hidden, ffn_ratio=1.0, dropout=0.0)
    out = model(span_reps=span_reps)
    print("Output shape:", out.shape)  # Expected (2, 9, 4)
    #print(out)

def test_between_context():
    print("\n=== Test Between-Context ===")
    batch, num_spans, hidden, seq_len = 2, 3, 4, 5
    span_reps = torch.randn(batch, num_spans, hidden)
    # Create dummy candidate span indices; here each span is defined as [start, end]
    span_ids = torch.tensor([[[0, 1], [1, 2], [2, 3]],
                                  [[0, 1], [2, 3], [4, 5]]])
    token_reps = torch.randn(batch, seq_len, hidden)
    token_masks = torch.ones(batch, seq_len, dtype=torch.bool)
    rel_masks = ~torch.eye(num_spans, dtype=torch.bool).unsqueeze(0).expand(batch, -1, -1)
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
    batch, num_spans, hidden, seq_len = 2, 3, 4, 5
    span_reps = torch.randn(batch, num_spans, hidden)
    span_ids = torch.tensor([[[0, 1], [1, 2], [2, 3]],
                                  [[0, 1], [2, 3], [3, 4]]])
    token_reps = torch.randn(batch, seq_len, hidden)
    token_masks = torch.ones(batch, seq_len, dtype=torch.bool)
    rel_masks = ~torch.eye(num_spans, dtype=torch.bool).unsqueeze(0).expand(batch, -1, -1)
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
    batch, num_spans, hidden, seq_len = 2, 3, 4, 5
    span_reps = torch.randn(batch, num_spans, hidden)
    span_ids = torch.tensor([[[0, 1], [1, 2], [2, 3]],
                                  [[0, 1], [2, 3], [2, 4]]])
    token_reps = torch.randn(batch, seq_len, hidden)
    token_masks = torch.ones(batch, seq_len, dtype=torch.bool)
    rel_masks = ~torch.eye(num_spans, dtype=torch.bool).unsqueeze(0).expand(batch, -1, -1)
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





