import torch
from torch import nn
import torch.nn.init as init

from .layers_other import FFNProjectionLayer


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


class RelRepNoContext(nn.Module):
    '''
    Constructs relation representations by concatenating the head and tail span representations
    and reprojecting them into a hidden space. This implementation does not use any contextual
    information beyond the span representations.

    Attributes:
        out_layer (nn.Module): A feedforward projection layer (FFNProjectionLayer) that takes 
                               concatenated head and tail representations and projects them into 
                               the desired hidden space.

    Args:
        hidden_size (int): Dimensionality of the input span representations.
        ffn_ratio (float): Expansion ratio for the intermediate layer in the feedforward network.
        dropout (float): Dropout rate applied in the feedforward network.
        **kwargs: Additional arguments (not used in this class).
    '''
    def __init__(self, hidden_size, ffn_ratio, dropout, **kwargs):
        super().__init__()
        self.out_layer = FFNProjectionLayer(input_dim  = hidden_size * 2,
                                            ffn_ratio  = ffn_ratio, 
                                            output_dim = hidden_size, 
                                            dropout    = dropout)
        
        #no weight init required as is handled by the out_layer

    def forward(self, cand_span_reps, **kwargs):
        '''
        Args:
            cand_span_reps (torch.Tensor): Tensor of shape (batch, top_k_spans, hidden) containing 
                                           span representations for candidate spans.
            **kwargs: Additional arguments (not used in this method).
        
        Returns:
            torch.Tensor: Tensor of shape (batch, top_k_spans**2, hidden), where each entry 
                          represents a relation representation for a pair of spans.
        '''
        batch, top_k_spans, _ = cand_span_reps.shape
        
        #Expand heads
        heads = cand_span_reps.unsqueeze(2).expand(-1, top_k_spans, top_k_spans, -1)  # (batch, top_k_spans, top_k_spans, hidden)
        #Expand tails
        tails = cand_span_reps.unsqueeze(1).expand(-1, top_k_spans, top_k_spans, -1)  # (batch, top_k_spans, top_k_spans, hidden)
        #make the rel reps
        rel_reps = torch.cat([heads, tails], dim=-1)   #(batch, top_k_spans, top_k_spans, hidden * 2)
        #Apply output layer
        rel_reps = self.out_layer(rel_reps)    #(batch, top_k_spans, top_k_spans, hidden)
        #move the shape back to 3 dims
        rel_reps = rel_reps.view(batch, top_k_spans * top_k_spans, -1)   #(batch, top_k_spans**2, hidden)

        return rel_reps    #(batch, top_k_spans**2, hidden)




class RelRepBetweenContext(nn.Module):
    '''
    Constructs relation representations by concatenating the head and tail span representations
    with a pooled context representation derived from the tokens between the spans. The resulting 
    representation is projected into a hidden space using an FFNProjectionLayer.

    The "between" context is derived as follows:
    - Tokens between the head and tail spans are identified based on their start and end indices.
    - If valid tokens exist between the spans, they are max-pooled to create the context representation.
    - If no valid tokens exist, the head span representation is used as the fallback.

    Attributes:
        out_layer (nn.Module): A feedforward projection layer (FFNProjectionLayer) that takes 
                               concatenated head, tail, and context representations and projects 
                               them into the desired hidden space.

    Args:
        hidden_size (int): Dimensionality of the input span representations.
        ffn_ratio (float): Expansion ratio for the intermediate layer in the feedforward network.
        dropout (float): Dropout rate applied in the feedforward network.
        **kwargs: Additional arguments (not used in this class).
    '''
    def __init__(self, hidden_size, ffn_ratio, dropout, **kwargs):
        super().__init__()
        self.out_layer = FFNProjectionLayer(input_dim = hidden_size * 3,
                                            ffn_ratio = ffn_ratio, 
                                            output_dim = hidden_size, 
                                            dropout = dropout)
        #no weight init required as is handled by the FFNLayer

    def forward(self, cand_span_reps, cand_span_ids, token_reps, token_masks, rel_masks, **kwargs):
        '''
        Args:
            cand_span_reps (torch.Tensor): Tensor of shape (batch, top_k_spans, hidden) containing 
                                           span representations for candidate spans.
            cand_span_ids (torch.Tensor): Tensor of shape (batch, top_k_spans, 2) containing 
                                          start and end token indices for candidate spans.
            token_reps (torch.Tensor): Tensor of shape (batch, seq_len, hidden) containing 
                                       token-level representations for the sequence.
            token_masks (torch.Tensor): Tensor of shape (batch, seq_len) containing a mask where 
                                        `True` indicates valid tokens.
            rel_masks: tensor fo shape (batch, top_k_spans**2) bool
                
        Returns:
            torch.Tensor: Tensor of shape (batch, top_k_spans**2, hidden), where each entry 
                          represents a relation representation for a pair of spans, including 
                          their concatenated head, tail, and context representations.
        '''
        batch, top_k_spans, hidden = cand_span_reps.shape

        #Generate head and tail representations
        head_reps = cand_span_reps.unsqueeze(2).expand(-1, top_k_spans, top_k_spans, -1)
        tail_reps = cand_span_reps.unsqueeze(1).expand(-1, top_k_spans, top_k_spans, -1)

        #Context reps
        ####################################################
        #Context calculation
        head_start_ids = cand_span_ids[:, :, 0].unsqueeze(2)  # (batch, top_k_spans, 1)
        head_end_ids = cand_span_ids[:, :, 1].unsqueeze(2)    # (batch, top_k_spans, 1)
        tail_start_ids = cand_span_ids[:, :, 0].unsqueeze(1)  # (batch, 1, top_k_spans)
        tail_end_ids = cand_span_ids[:, :, 1].unsqueeze(1)    # (batch, 1, top_k_spans)

        #Compute "between" bounds
        min_end = torch.min(head_end_ids, tail_end_ids)
        max_start = torch.max(head_start_ids, tail_start_ids)
        # Mask to identify valid "between" tokens
        valid_between = (min_end < max_start) & rel_masks.view(batch, top_k_spans, top_k_spans)
        #Create expanded masks for indexing
        seq_indices = torch.arange(token_reps.shape[1], device=token_reps.device).unsqueeze(0).expand(batch, -1)   # (batch, seq_len)
        #Create token masks for "between" ranges
        expanded_min_end = min_end.unsqueeze(-1)  # (batch, top_k_spans, top_k_spans, 1)
        expanded_max_start = max_start.unsqueeze(-1)  # (batch, top_k_spans, top_k_spans, 1)
        expanded_valid_between = valid_between.unsqueeze(-1)  # (batch, top_k_spans, top_k_spans, 1)
        #Get boolean masks for tokens in valid "between" ranges
        combined_context_mask = (seq_indices.unsqueeze(1).unsqueeze(1) >= expanded_min_end) & \
                              (seq_indices.unsqueeze(1).unsqueeze(1) < expanded_max_start) & \
                              expanded_valid_between # (batch, top_k_spans, top_k_spans, seq_len)

        # Include global token validity in the mask
        valid_token_mask = token_masks.unsqueeze(1).unsqueeze(1)
        combined_context_mask = combined_context_mask & valid_token_mask
        #Apply the token masks to token representations, broadcast and maxpool
        context_reps, _ = token_reps.unsqueeze(1).unsqueeze(1).masked_fill(~combined_context_mask.unsqueeze(-1), float('-inf')).max(dim=3)  # (batch, top_k_spans, top_k_spans, hidden)
        ####################################################

        # Concatenate head, tail, and context representations
        rel_reps = torch.cat([head_reps, tail_reps, context_reps], dim=-1)  # (batch, top_k_spans, top_k_spans, hidden*3)

        # Apply output layer and reshape
        rel_reps = self.out_layer(rel_reps)     # (batch, top_k_spans, top_k_spans, hidden)
        rel_reps = rel_reps.view(batch, top_k_spans * top_k_spans, -1)  # (batch, top_k_spans**2, hidden)

        return rel_reps




class RelRepWindowContext(nn.Module):
    '''
    Constructs relation representations by concatenating the head and tail span representations
    with a pooled context representation derived from the tokens around the spans based on a window size.
    The resulting representation is projected into a hidden space using an FFNProjectionLayer.

    The "window" context is derived by selecting tokens within a specified window before and after the span:
    - For each span, the context before and after within the window size is pooled (max or average based on the configuration).
    - Tokens within the span are explicitly excluded from the context window.
    - Tokens masked out by the token mask are set to -inf prior to pooling to exclude them from contributing to the pooled output.

    Attributes:
        out_layer (nn.Module): A feedforward projection layer (FFNProjectionLayer) that projects the concatenated
                               head, tail, and context representations into the desired hidden space.
        window_size (int): The number of tokens to consider before and after the span for context pooling.

    Args:
        hidden_size (int): Dimensionality of the input span representations.
        ffn_ratio (float): Expansion ratio for the intermediate layer in the feedforward network.
        dropout (float): Dropout rate applied in the feedforward network.
        window_size (int): Size of the window for context tokens around the spans.
    '''
    def __init__(self, hidden_size, ffn_ratio, dropout, window_size):
        super().__init__()
        self.window_size = window_size
        self.out_layer = FFNProjectionLayer(input_dim = hidden_size * 3,
                                            ffn_ratio = ffn_ratio, 
                                            output_dim = hidden_size, 
                                            dropout = dropout)
        

    def forward(self, cand_span_reps, cand_span_ids, token_reps, token_masks, **kwargs):
        batch, top_k_spans, hidden = cand_span_reps.shape  # (batch, top_k_spans, hidden)

        # Span representation expansion for relational context
        head_reps = cand_span_reps.unsqueeze(2).expand(-1, top_k_spans, top_k_spans, -1)  # (batch, top_k_spans, top_k_spans, hidden)
        tail_reps = cand_span_reps.unsqueeze(1).expand(-1, top_k_spans, top_k_spans, -1)  # (batch, top_k_spans, top_k_spans, hidden)
        
        #Context reps
        ####################################################
        # Calculating window indices
        head_start_ids = cand_span_ids[:, :, 0].unsqueeze(2)  # (batch, top_k_spans, 1)
        head_end_ids = cand_span_ids[:, :, 1].unsqueeze(2)    # (batch, top_k_spans, 1)
        tail_start_ids = cand_span_ids[:, :, 0].unsqueeze(1)  # (batch, 1, top_k_spans)
        tail_end_ids = cand_span_ids[:, :, 1].unsqueeze(1)    # (batch, 1, top_k_spans)

        # Expanding token indices for window context
        seq_indices = torch.arange(token_reps.shape[1], device=token_reps.device).unsqueeze(0).expand(batch, -1)   # (batch, seq_len)

        # Window masks for head and tail spans, excluding the span itself
        head_mask = ((seq_indices.unsqueeze(1).unsqueeze(1) >= (head_start_ids - self.window_size)) & \
                    (seq_indices.unsqueeze(1).unsqueeze(1) < head_start_ids)) | \
                    ((seq_indices.unsqueeze(1).unsqueeze(1) > head_end_ids) & \
                    (seq_indices.unsqueeze(1).unsqueeze(1) <= (head_end_ids + self.window_size)))  # (batch, top_k_spans, top_k_spans, seq_len)

        tail_mask = ((seq_indices.unsqueeze(1).unsqueeze(1) >= (tail_start_ids - self.window_size)) & \
                    (seq_indices.unsqueeze(1).unsqueeze(1) < tail_start_ids)) | \
                    ((seq_indices.unsqueeze(1).unsqueeze(1) > tail_end_ids) & \
                    (seq_indices.unsqueeze(1).unsqueeze(1) <= (tail_end_ids + self.window_size)))  # (batch, top_k_spans, top_k_spans, seq_len)

        # Combine the head and tail masks to get a single mask for the context
        combined_context_mask = head_mask | tail_mask  # (batch, top_k_spans, top_k_spans, seq_len)

        # Include global token validity in the mask
        valid_token_mask = token_masks.unsqueeze(1).unsqueeze(1)
        combined_context_mask = combined_context_mask & valid_token_mask    # (batch, top_k_spans, top_k_spans, seq_len)

        #this expands the token reps to (batch, top_k_spans, top_k_spans, seq_len, hidden)
        #then applies the rel_specific context mask to set masked out tokens to -inf
        #then max pools over the seq dim (dim 3)
        context_reps, _ =   token_reps.unsqueeze(1).unsqueeze(1).masked_fill(~combined_context_mask.unsqueeze(-1), float('-inf')).max(dim=3)  # (batch, top_k_spans, top_k_spans, hidden)
        ####################################################

        # Concatenate head, tail, and combined context representations
        rel_reps = torch.cat([head_reps, tail_reps, context_reps], dim=-1)  # (batch, top_k_spans, top_k_spans, 3 * hidden)

        # Apply output layer and reshape
        rel_reps = self.out_layer(rel_reps)     # (batch, top_k_spans, top_k_spans, hidden)
        rel_reps = rel_reps.view(batch, top_k_spans * top_k_spans, -1)  # (batch, top_k_spans**2, hidden)

        return rel_reps        



class RelationRepLayer(nn.Module):
    """
    Make the relation reps, has several options, based on config.rel_mode:
    1) no_context => graphER => juts concat the head and tail span reps
    2) between_context => concatenate the head and tail reps with between spans rep for context => to be coded
    3) window_context => concatenate the head and tail reps with windowed context reps, i.e. window context takes token reps in a window before and after each span and attention_pools, max_pool them => to be coded

    The init call from models....
        self.rel_rep_layer = RelationRepLayer(
            #specifically named....
            rel_mode    = config.rel_mode,    #what kind of rel_rep generation algo to use 

            #the rest are in kwargs...
            hidden_size = config.hidden_size, #the hidden size coming in and going out 
            ffn_ratio   = config.ffn_ratio,
            dropout     = config.dropout 

            ...
        )
    """
    def __init__(self, rel_mode, **kwargs):
        super().__init__()
        #kwargs has remaining: []
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
        '''
        rel_reps = self.rel_rep_layer(cand_span_reps, 
                                      cand_span_ids, 
                                      token_reps, 
                                      token_masks)   
        '''
        result = self.rel_rep_layer(**kwargs)

        return result







#Trying to get this test code working
#Trying to get this test code working
#Trying to get this test code working
#Trying to get this test code working
#Trying to get this test code working
#Trying to get this test code working
#Trying to get this test code working

def test_relation_rep_layers():
    # Define parameters
    batch_size = 2
    top_k_spans = 4
    seq_len = 20
    hidden_size = 4
    ffn_ratio = 2.0
    dropout = 0.1

    # Generate random inputs
    cand_span_reps = torch.randn(batch_size, top_k_spans, hidden_size)
    cand_span_ids = torch.randint(0, seq_len, (batch_size, top_k_spans, 2))  # Random start and end indices
    cand_span_ids[:, :, 1] += 1  # Ensure end indices are greater than start indices
    token_reps = torch.randn(batch_size, seq_len, hidden_size)
    token_masks = torch.randint(0, 2, (batch_size, seq_len), dtype=torch.bool)  # Random binary mask
    rel_masks = torch.ones((batch_size, top_k_spans * top_k_spans), dtype=torch.bool)  # Random binary mask


    #force set the token mask and span_ids so it makes sense
    token_masks = [[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0], [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0]]
    token_masks = torch.tensor(token_masks, dtype=torch.bool)  # Random binary mask
    cand_span_ids = [[[4,8],
                     [10,15],
                     [0,2],
                     [0,0]],
                     [[2,6],
                     [8,15],
                     [1,5],
                     [12,16]]]
    cand_span_ids = torch.tensor(cand_span_ids, dtype=torch.int)



    print(cand_span_ids)
    print(token_masks)


    # Initialize models
    rel_no_context = RelRepNoContext(hidden_size, ffn_ratio, dropout)
    rel_between_context_old = RelRepBetweenContext_old(hidden_size, ffn_ratio, dropout)
    rel_between_context = RelRepBetweenContext(hidden_size, ffn_ratio, dropout)

    # Test RelRepNoContext
    print("Testing RelRepNoContext...")
    rel_no_context_output = rel_no_context(cand_span_reps)
    assert rel_no_context_output.shape == (batch_size, top_k_spans**2, hidden_size), \
        f"Unexpected output shape: {rel_no_context_output.shape}"
    print(f"Output shape: {rel_no_context_output.shape}")

    # Test RelRepBetweenContext_old
    print("\nTesting RelRepBetweenContext_old...")
    rel_between_context_old_output = rel_between_context_old(
        cand_span_reps, cand_span_ids, token_reps, token_masks, rel_masks
    )
    assert rel_between_context_old_output.shape == (batch_size, top_k_spans**2, hidden_size), \
        f"Unexpected output shape: {rel_between_context_old_output.shape}"
    print(f"Output shape: {rel_between_context_old_output.shape}")

    # Test RelRepBetweenContext
    print("\nTesting RelRepBetweenContext...")
    rel_between_context_output = rel_between_context(
        cand_span_reps, cand_span_ids, token_reps, token_masks, rel_masks
    )
    assert rel_between_context_output.shape == (batch_size, top_k_spans**2, hidden_size), \
        f"Unexpected output shape: {rel_between_context_output.shape}"
    print(f"Output shape: {rel_between_context_output.shape}")

    print('NOTE: to test equivalence, you need to disable the output layer')



if __name__ == "__main__":
    test_relation_rep_layers()


