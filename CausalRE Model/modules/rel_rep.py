import torch
from torch import nn
import torch.nn.init as init

#from .layers_other import FFNProjectionLayer
from layers_other import FFNProjectionLayer





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
        #no weight init required as is handled by the FFNLayer

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



class RelRepBetweenContext_old(nn.Module):
    '''
    adds in betwn span context
    '''
    def __init__(self, hidden_size, ffn_ratio, dropout, **kwargs):
        super().__init__()
        self.out_layer = FFNProjectionLayer(input_dim = hidden_size * 3,
                                            ffn_ratio = ffn_ratio, 
                                            output_dim = hidden_size, 
                                            dropout = dropout)

    def forward(self, cand_span_reps, cand_span_ids, token_reps, token_masks, **kwargs):
        '''
        - cand_span_reps: Span representations, shape (batch, top_k_spans, hidden)
        - cand_span_ids: Token indices for spans, shape (batch, top_k_spans, 2)
        - token_reps: Token representations, shape (batch, seq_len, hidden)
        - token_masks: Token masks, shape (batch, seq_len)

        Returns:
        - rel_reps: Relation representations, shape (batch, top_k_spans**2, hidden)
        '''
        batch, top_k_spans, hidden = cand_span_reps.shape

        #Head and tail representations => easy
        #Expand heads
        heads = cand_span_reps.unsqueeze(2).expand(-1, top_k_spans, top_k_spans, -1)  # (batch, top_k_spans, top_k_spans, hidden)
        #Expand tails
        tails = cand_span_reps.unsqueeze(1).expand(-1, top_k_spans, top_k_spans, -1)  # (batch, top_k_spans, top_k_spans, hidden)
        #make the rel reps
        rel_reps = torch.cat([heads, tails], dim=-1)   #(batch, top_k_spans, top_k_spans, hidden * 2)

        #get the context reps => rather involved
        #Extract span boundaries
        head_start_ids = cand_span_ids[:, :, 0].unsqueeze(2)  # (batch, top_k_spans, 1)
        head_end_ids = cand_span_ids[:, :, 1].unsqueeze(2)    # (batch, top_k_spans, 1)
        tail_start_ids = cand_span_ids[:, :, 0].unsqueeze(1)  # (batch, 1, top_k_spans)
        tail_end_ids = cand_span_ids[:, :, 1].unsqueeze(1)    # (batch, 1, top_k_spans)
        #Compute "between" bounds
        min_end = torch.min(head_end_ids, tail_end_ids)  # (batch, top_k_spans, top_k_spans)
        max_start = torch.max(head_start_ids, tail_start_ids)  # (batch, top_k_spans, top_k_spans)
        #Mask to identify valid "between" tokens
        valid_between = min_end < max_start # (batch, top_k_spans, top_k_spans)
        # Create a tensor for context representations
        context_reps = torch.zeros(batch, top_k_spans, top_k_spans, hidden, device=cand_span_reps.device)
        #now fill the tensor by nested looping => will remove later
        for b in range(batch):
            for i in range(top_k_spans):
                for j in range(top_k_spans):
                    if valid_between[b, i, j]:
                        #Determine the between token range
                        start_idx = min_end[b, i, j].item()
                        end_idx = max_start[b, i, j].item()
                        #Extract tokens in the "between" region
                        between_tokens = token_reps[b, start_idx:end_idx, :]  # (seq_len_between, hidden)
                        between_mask = token_masks[b, start_idx:end_idx]      # (seq_len_between)
                        # Apply mask: set masked positions to -inf
                        between_tokens = between_tokens.masked_fill(~between_mask.unsqueeze(-1), float('-inf'))
                        # Max-pool over the "between" tokens
                        pooled_context, _ = torch.max(between_tokens, dim=0)  # (hidden,)
                        context_reps[b, i, j, :] = pooled_context
                    else:
                        # If no valid "between" tokens, use the head span representation
                        context_reps[b, i, j, :] = cand_span_reps[b, i, :]

        # Concatenate head, tail, and context representations
        rel_reps = torch.cat([rel_reps, context_reps], dim=-1)  # (batch, top_k_spans, top_k_spans, hidden*3)

        # Apply output layer
        rel_reps = self.out_layer(rel_reps)
        rel_reps = rel_reps.view(batch, top_k_spans * top_k_spans, -1)  # (batch, top_k_spans**2, hidden)

        return rel_reps




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

    def forward(self, cand_span_reps, cand_span_ids, token_reps, token_masks, **kwargs):
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

        Returns:
            torch.Tensor: Tensor of shape (batch, top_k_spans**2, hidden), where each entry 
                          represents a relation representation for a pair of spans, including 
                          their concatenated head, tail, and context representations.
        '''
        batch, top_k_spans, hidden = cand_span_reps.shape

        #Head and tail reps => easy
        #Expand heads
        heads = cand_span_reps.unsqueeze(2).expand(-1, top_k_spans, top_k_spans, -1)  # (batch, top_k_spans, top_k_spans, hidden)
        #Expand tails
        tails = cand_span_reps.unsqueeze(1).expand(-1, top_k_spans, top_k_spans, -1)  # (batch, top_k_spans, top_k_spans, hidden)
        #make the rel reps
        rel_reps = torch.cat([heads, tails], dim=-1)   #(batch, top_k_spans, top_k_spans, hidden * 2)

        #Context reps => rather involved
        #Initialize context representations
        context_reps = cand_span_reps.unsqueeze(2).expand(-1, -1, top_k_spans, -1)  # (batch, top_k_spans, top_k_spans, hidden)

        #Extract span boundaries
        head_start_ids = cand_span_ids[:, :, 0].unsqueeze(2)  # (batch, top_k_spans, 1)
        head_end_ids = cand_span_ids[:, :, 1].unsqueeze(2)    # (batch, top_k_spans, 1)
        tail_start_ids = cand_span_ids[:, :, 0].unsqueeze(1)  # (batch, 1, top_k_spans)
        tail_end_ids = cand_span_ids[:, :, 1].unsqueeze(1)    # (batch, 1, top_k_spans)
        #Compute "between" bounds
        min_end = torch.min(head_end_ids, tail_end_ids)  # (batch, top_k_spans, top_k_spans)
        max_start = torch.max(head_start_ids, tail_start_ids)  # (batch, top_k_spans, top_k_spans)
        # Mask to identify valid "between" tokens
        valid_between = min_end < max_start  # (batch, top_k_spans, top_k_spans)

        #Create expanded masks for indexing
        seq_indices = torch.arange(token_reps.shape[1], device=cand_span_reps.device)
        seq_indices = seq_indices.unsqueeze(0).expand(batch, -1)  # (batch, seq_len)
        #Create token masks for "between" ranges
        expanded_min_end = min_end.unsqueeze(-1)  # (batch, top_k_spans, top_k_spans, 1)
        expanded_max_start = max_start.unsqueeze(-1)  # (batch, top_k_spans, top_k_spans, 1)
        #Get boolean masks for tokens in valid "between" ranges
        token_mask_in_range = (seq_indices.unsqueeze(1).unsqueeze(1) >= expanded_min_end) & \
                              (seq_indices.unsqueeze(1).unsqueeze(1) < expanded_max_start)  # (batch, top_k_spans, top_k_spans, seq_len)
        #Apply the token masks to token representations
        masked_token_reps = token_reps.unsqueeze(1).unsqueeze(1)
        masked_token_reps = masked_token_reps.masked_fill(~token_mask_in_range.unsqueeze(-1), float('-inf'))  # (batch, top_k_spans, top_k_spans, seq_len, hidden)
        # Perform max pooling over the "between" tokens
        pooled_context, _ = masked_token_reps.max(dim=3)  # (batch, top_k_spans, top_k_spans, hidden)

        #Overwrite context representations for valid "between" regions
        context_reps = torch.where(valid_between.unsqueeze(-1), pooled_context, context_reps)

        # Concatenate head, tail, and context representations
        rel_reps = torch.cat([rel_reps, context_reps], dim=-1)  # (batch, top_k_spans, top_k_spans, hidden*3)
        # Apply output layer
        rel_reps = self.out_layer(rel_reps)
        rel_reps = rel_reps.view(batch, top_k_spans * top_k_spans, -1)  # (batch, top_k_spans**2, hidden)

        return rel_reps





class RelRepWindowContext(nn.Module):
    '''
    adds in window context ni window before/after each span
    '''
    def __init__(self, hidden_size, ffn_ratio, dropout, pooling, **kwargs):
        super().__init__()
        self.pooling = pooling
        self.out_layer = FFNProjectionLayer(input_dim = hidden_size * 3,
                                            ffn_ratio = ffn_ratio, 
                                            output_dim = hidden_size, 
                                            dropout = dropout)

        self.init_weights()


    def init_weights(self):
        init.xavier_normal_(self.out_layer.weight)
        init.constant_(self.out_layer.bias, 0)


    def forward(self, span_reps, span_ids, token_reps, token_ids, pooling, **kwargs):
        '''
        describe it


        again get the context reps and concat with the head and tail reps
        '''
        raise Exception('have not done this code yet')

        batch, top_k_spans, _ = span_reps.shape
        
        #do heads
        heads = span_reps
        #Apply projection layer to reduce hidden dims by half for concatenation
        heads = self.head_layer(heads)
        #Expand heads to create relation representations
        heads = heads.unsqueeze(2)   #(batch, top_k_spans, 1, hidden//2)

        #do tails
        tails = span_reps
        #Apply projection layer to reduce hidden dims by half for concatenation
        tails = self.tail_layer(tails)
        #Expand tails to create relation representations
        tails = tails.unsqueeze(1)   #(batch, 1, top_k_spans, hidden//2)
        
        #make the rel reps
        rel_reps = torch.cat([heads, tails], dim=-1)   #(batch, top_k_spans, top_k_spans, hidden)
        # Apply output layer
        rel_reps = self.out_layer(rel_reps)
        #move the shape back to 3 dims
        rel_reps = rel_reps.view(batch, top_k_spans * top_k_spans, -1)   

        return rel_reps    #(batch, top_k_spans**2, hidden)









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
    top_k_spans = 5
    seq_len = 20
    hidden_size = 16
    ffn_ratio = 2.0
    dropout = 0.1

    # Generate random inputs
    cand_span_reps = torch.randn(batch_size, top_k_spans, hidden_size)
    cand_span_ids = torch.randint(0, seq_len, (batch_size, top_k_spans, 2))  # Random start and end indices
    cand_span_ids[:, :, 1] += 1  # Ensure end indices are greater than start indices
    token_reps = torch.randn(batch_size, seq_len, hidden_size)
    token_masks = torch.randint(0, 2, (batch_size, seq_len), dtype=torch.bool)  # Random binary mask

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
        cand_span_reps, cand_span_ids, token_reps, token_masks
    )
    assert rel_between_context_old_output.shape == (batch_size, top_k_spans**2, hidden_size), \
        f"Unexpected output shape: {rel_between_context_old_output.shape}"
    print(f"Output shape: {rel_between_context_old_output.shape}")

    # Test RelRepBetweenContext
    print("\nTesting RelRepBetweenContext...")
    rel_between_context_output = rel_between_context(
        cand_span_reps, cand_span_ids, token_reps, token_masks
    )
    assert rel_between_context_output.shape == (batch_size, top_k_spans**2, hidden_size), \
        f"Unexpected output shape: {rel_between_context_output.shape}"
    print(f"Output shape: {rel_between_context_output.shape}")




if __name__ == "__main__":
    test_relation_rep_layers()


