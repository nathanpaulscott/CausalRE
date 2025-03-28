import math, random, os
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import torch.nn.init as init

from .layers_other import PositionalEncoding, ProjectionLayer

#################################################################
#################################################################
#################################################################

class First_n_Last_graphER(nn.Module):
    '''
    orig comment
    Marks and projects span endpoints using an MLP.
    
    Nathan
    This was from grpahER, they 
    - run token_reps through a FFN first separatly for the reps used for the start reps and the end reps, it is not clear this will have any use!!
    - then extract the span_start and end reps from token reps using the candidate span_ids
    - then concatenate the start and end reps together for each candidate span making the last dim hidden*2
    - then reproject the concatenated span_reps back to hidden

    outputs:
        span_reps, a tensor of shape (batch, num_spans, hidden) where num_spans = seq_len*max_span_width
    
    NOTE: span_ids with start, end = 0,0 are processed nromally, but the reps will be ignored later by the span_masks
   '''
    def __init__(self, 
                 max_span_width, 
                 hidden_size, 
                 layer_type,
                 ffn_ratio,
                 dropout, 
                 **kwargs):
        super().__init__()
        #overwrite the passed values to copy graphER
        ffn_ratio = 1.5
        dropout = 0.4
        self.max_span_width = max_span_width     #max span width
        #self.project_start = ProjectionLayer(hidden_size, hidden_size, dropout, layer_type, ffn_ratio)     #FFN for the start token    => no point in this, I have removed it
        #self.project_end = ProjectionLayer(hidden_size, hidden_size, dropout, layer_type, ffn_ratio)       #FFN for the end token      => no point in this, I have removed it
        self.out_project = ProjectionLayer(2*hidden_size, hidden_size, dropout, layer_type, ffn_ratio)     #projects the concat of the start/end back to hidden
        self.relu = nn.ReLU()
        #init weights not required as is handled by FFNLayers

    def forward(self, token_reps, span_ids, span_masks, **kwargs):
        '''
        token_reps is of shape  (batch, seq_len, hidden)    where seq_len is w_seq_len for word token aligned token_reps
        span_ids is of shape    (batch, num_spans, 2)       where num_spans = w_seq_len*max_span_width, values in last dim are w/sw aligned based on pooling
        span_masks is of shape   (batch, num_spans)          where num_spans = w_seq_len*max_span_width
        Because when they generate all possible spans they just generate all spans from width 1 to max_span_width for each start idx, even when the end idx is out of the max_seq_len, they just mask those as invalid spans 
        So that view statement at the end will work ok, it just kind of associates the max_span_width spans with each start token idx in a way
        '''
        #Zero out invalid indices - they'll get token_reps[0], which will be ignored later
        span_ids = span_ids * span_masks

        #run the token_reps through a simple FFN for the start and end token
        #it actually does nothing, but further mix the token embeddings around
        #is this really neccessary?
        #start_token_reps = self.project_start(token_reps)
        #end_token_reps = self.project_end(token_reps)
        start_token_reps = token_reps
        end_token_reps = token_reps
        start_ids = span_ids[:, :, 0]
        end_ids = span_ids[:, :, 1] - 1
        #extract the start rep and end rep out of token_reps
        start_span_rep = extract_rep(start_token_reps, ids=start_ids)
        end_span_rep = extract_rep(end_token_reps, ids=end_ids)
        #make the span reps
        span_reps = torch.cat([start_span_rep, end_span_rep], dim=-1)
        span_reps = self.relu(span_reps)
        span_reps = self.out_project(span_reps)
        return span_reps    #(batch, num_spans, hidden)

#################################################################
#################################################################
#################################################################

class First_n_Last(nn.Module):
    '''
    Simplified version of the first and last idea from graphER
    It doesn't use any FFN layers, just extracts and concats the start and end token reps for each span
    and re-projects back to hidden with a simple reprojection layer

    outputs:
        span_reps, a tensor of shape (batch, num_spans, hidden) 
            where num_spans = word_token_seq_len*max_span_width
            NOTE: if token_reps are sw aligned, seq_len is sw_token_seq_len!!!  Remember that

    NOTE: span_ids with start, end = 0,0 are processed normally, but the reps will be ignored later by the span_masks
   '''
    def __init__(self, max_span_width, hidden_size, dropout, layer_type, ffn_ratio, **kwargs):
        super().__init__()
        #self.out_project = nn.Linear(2*hidden_size, hidden_size)    #projects the concat of the start/end back to hidden
        self.out_project = ProjectionLayer(2*hidden_size, hidden_size, dropout, layer_type, ffn_ratio)     #projects the concat of the start/end back to hidden


    def forward(self, token_reps, span_ids, span_masks, neg_limit=None, **kwargs):
        '''
        token_reps is of shape  (batch, seq_len, hidden)    => w_seq_len for word token aligned, or sw_seq_len for sw token aligned, it matters!!
        span_ids is of shape    (batch, num_spans, 2)       where num_spans = w_seq_len*max_span_width, values in last dim are w/sw aligned based on pooling
        span_masks is of shape   (batch, num_spans)          where num_spans = w_seq_len*max_span_width
        '''
        #extract the start and end span reps
        span_reps = extract_span_reps(token_reps, 
                                      span_ids, 
                                      span_masks, 
                                      mode='start_end',
                                      neg_limit = neg_limit)
        span_reps = self.out_project(span_reps)
        return span_reps   #shape (batch, num_spans, hidden)



#################################################################
#################################################################
#################################################################

class Spert(nn.Module):
    '''
    The strategy from Spert, can operate on word token_reps or sw token_reps, but only on HF models as it requires cls_reps
    inputs:
        token_reps => word or sw alignment
        cls_reps
        width_embeddings => the span width embeddings (if sw token alignment, these are embeddings for sw token span width, if word token alignment these are embeddings for word token span width)
    outputs:
        span_reps, a tensor of shape (batch, num_spans, hidden) where num_spans = seq_len*max_span_width

    NOTE: span_ids with start, end = 0,0 are processed nromally, but the reps will be ignored later by the span_masks
    '''    
    def __init__(self, 
                 max_span_width, 
                 hidden_size, 
                 width_embeddings, 
                 cls_flag,
                 dropout,
                 layer_type,
                 ffn_ratio,
                 **kwargs):
        super().__init__()
        #kwargs has remaining: ['use_span_pos_encoding']
        self.max_span_width = max_span_width
        self.width_embeddings = width_embeddings
        internal_hidden_size = hidden_size + width_embeddings.embedding_dim
        self.cls_flag = cls_flag
        if self.cls_flag:
            internal_hidden_size += hidden_size

        #self.out_project = nn.Linear(internal_hidden_size, hidden_size)
        self.out_project = ProjectionLayer(internal_hidden_size, hidden_size, dropout, layer_type, ffn_ratio)    


    def forward(self, token_reps, span_ids, span_masks, cls_reps=None, span_widths=None, neg_limit=None, **kwargs):
        '''
        token_reps: (batch, seq_len, hidden)
        span_ids is of shape    (batch, num_spans, 2)       where num_spans = w_seq_len*max_span_width, values in last dim are w/sw aligned based on pooling
        span_masks is of shape   (batch, num_spans)          where num_spans = w_seq_len*max_span_width
        cls_reps: (batch, hidden)
        span_widths => the span word token widths used for width embeddings
        '''
        num_spans = span_ids.shape[1]
        # Get maxpooled span representations
        span_maxpool_reps = extract_span_reps(token_reps, 
                                              span_ids, 
                                              span_masks, 
                                              mode='maxpool',
                                              neg_limit = neg_limit)
        # Get width embeddings
        width_emb = self.width_embeddings(span_widths)
        # Combine components
        span_reps = torch.cat([span_maxpool_reps, width_emb], dim=-1)
        #add cls reps if required
        if self.cls_flag and cls_reps is not None:
            #Expand cls_reps
            cls_expanded = cls_reps.unsqueeze(1).expand(-1, num_spans, -1)
            span_reps = torch.cat([cls_expanded, span_reps], dim=-1)

        span_reps = self.out_project(span_reps)
        return span_reps    #(batch, num_spans, hidden)


#################################################################
#################################################################
#################################################################

class Nathan(nn.Module):
    '''
    Here we concat 5 things:
    - first word token rep
    - maxpool of all the span reps between the first and end token rep (if there are no internal tokens, then this will be a repeat of the start token)
    - end word token rep (if the span is one token long, then this will eb a repeat of the start token)
    - width embedding rep
    - cls token rep if cls_flag is True

    NOTE: span_ids with start, end = 0,0 are processed normally, but the reps will be ignored later by the span_masks
    '''    
    def __init__(self, 
                 max_span_width, 
                 hidden_size, 
                 width_embeddings, 
                 cls_flag,
                 dropout,
                 layer_type,
                 ffn_ratio,
                 **kwargs):
        '''
        hidden size is the model hidden size
        max_span_width is the max span width in word tokens from the model configs
        width embeddings are the span word token width embeddings from the model init
        cls_flag indicates whether a cls token rep is available and to be used
        '''
        super().__init__()
        self.max_span_width = max_span_width
        self.width_embeddings = width_embeddings
        internal_hidden_size = 3*hidden_size + width_embeddings.embedding_dim
        self.cls_flag = cls_flag
        if self.cls_flag:
            internal_hidden_size += hidden_size

        self.out_project = ProjectionLayer(internal_hidden_size, hidden_size, dropout, layer_type, ffn_ratio)    



    def forward(self, token_reps, span_ids, span_masks, cls_reps=None, span_widths=None, neg_limit=None, alpha=1, **kwargs):
        '''
        token_reps: (batch, seq_len, hidden)
        span_ids is of shape    (batch, num_spans, 2)       where num_spans = w_seq_len*max_span_width, values in last dim are w/sw aligned based on pooling
        span_masks is of shape   (batch, num_spans)          where num_spans = w_seq_len*max_span_width
        cls_reps: (batch, hidden)
        span_widths => the span word token widths used for width embeddings

        NOTE: for span_ids as end are python list style (actual + 1), end is always > start.
        edge cases:
        - span of width 0   => invalid span with start/end = 0, give all 0s for span rep
        - span of width 1   => internal dim = start_rep*3 + width_emb + [cls_rep]
        - span of width 2   => internal dim = start_rep*2 + end_rep  + width_emb + [cls_rep]
        - span of width > 2 => internal dim = start_rep + maxpool_inner_rep + end_rep  + width_emb + [cls_rep]
        '''
        num_spans = span_ids.shape[1]
        #extract the span_reps as start + inner_maxpool + end
        span_reps = extract_span_reps(token_reps, 
                                      span_ids, 
                                      span_masks, 
                                      mode = 'start_inner_maxpool_end', 
                                      neg_limit = neg_limit,
                                      alpha = alpha)
        # Get width embeddings
        width_emb = self.width_embeddings(span_widths)
        #Combine Components
        span_reps = torch.cat([span_reps, width_emb], dim=-1)

        if self.cls_flag and cls_reps is not None:
            #Expand cls_reps
            cls_expanded = cls_reps.unsqueeze(1).expand(-1, num_spans, -1)
            span_reps = torch.cat([span_reps, cls_expanded], dim=-1)

        span_reps = self.out_project(span_reps)
        return span_reps




class Attn(nn.Module):
    '''
    NOTE: span_ids with start, end = 0,0 are processed normally, but the reps will be ignored later by the span_masks
    '''    
    def __init__(self, 
                 max_span_width, 
                 hidden_size, 
                 width_embeddings, 
                 cls_flag,
                 dropout,
                 layer_type,
                 ffn_ratio,
                 **kwargs):
        '''
        hidden size is the model hidden size
        max_span_width is the max span width in word tokens from the model configs
        width embeddings are the span word token width embeddings from the model init
        cls_flag indicates whether a cls token rep is available and to be used
        '''
        super().__init__()
        self.max_span_width = max_span_width
        self.width_embeddings = width_embeddings
        internal_hidden_size = hidden_size + width_embeddings.embedding_dim
        self.cls_flag = cls_flag
        if self.cls_flag:
            internal_hidden_size += hidden_size
        self.out_project = ProjectionLayer(internal_hidden_size, hidden_size, dropout, layer_type, ffn_ratio)    
        self.attention_layer = nn.Linear(hidden_size, 1)
        self.init_weights()

    def init_weights(self):
        torch.manual_seed(42)
        init.xavier_uniform_(self.attention_layer.weight)
        if self.attention_layer.bias is not None:
            init.constant_(self.attention_layer.bias, 0)

    def attn_pooler(self, token_reps, span_starts, span_ends, span_masks, neg_limit):
        """
        Performs attention-based pooling on token representations within specified spans, considering only valid tokens.

        This method aggregates token representations within each span into a single vector using a learned attention mechanism. 
        It ensures that only valid tokens (as indicated by span_masks and within span boundaries) contribute to the final span representation.

        Parameters:
            token_reps (Tensor): The token representations with shape [batch, 1, seq_len, hidden_dim].
                This tensor should contain the embeddings of all tokens across all sequences in the batch.
            span_starts (Tensor): The starting indices of spans with shape [batch, num_spans].
                Indicates the start of each span to be pooled.
            span_ends (Tensor): The ending indices of spans with shape [batch, num_spans], inclusive.
                Indicates the end of each span to be pooled.
            span_masks (Tensor): A boolean tensor with shape [batch, num_spans] that indicates whether each span is valid.
                This mask helps in ignoring spans that should not be processed due to padding or other factors.
            neg_limit (float): A large negative value used during softmax to mask out invalid tokens effectively.
                This value should be significantly negative to ensure these entries do not affect the softmax output.

        Returns:
            Tensor: The aggregated span representations with shape [batch, num_spans, hidden_dim].
                Each span's representation is a weighted sum of its token embeddings, where the weights are determined by an attention mechanism.

        Notes:
            - The method handles variable span widths by dynamically adjusting the masks and processing only the valid portions of each span.
            - It supports batches with variable numbers of spans and different levels of validity across the spans.
        """        
        # Assuming token_reps shape is [batch, 1, seq_len, hidden]
        batch_size, _, seq_len, hidden_dim = token_reps.shape
        span_widths = span_ends - span_starts + 1
        max_span_width = (span_widths).max()
        token_within_span_masks = torch.arange(max_span_width, device=token_reps.device).unsqueeze(0).unsqueeze(0) < (span_widths).unsqueeze(-1)
        # Ensure that token_within_span_masks are further restricted by span_masks
        valid_token_masks = token_within_span_masks & span_masks.unsqueeze(-1)
        # Gather inputs for all spans: shape [batch_size, num_spans, max_span_width, hidden_dim]
        span_reps = torch.zeros(batch_size, span_starts.size(1), max_span_width, hidden_dim, device=token_reps.device)
        for b in range(batch_size):
            for i in range(span_starts.size(1)):
                span_width = span_widths[b, i]
                span_reps[b, i, :span_width] = token_reps[b, 0, span_starts[b, i]:span_ends[b, i] + 1]

        # Apply the attention layer and compute weights
        flat_span_reps = span_reps.view(-1, max_span_width, hidden_dim)  # Flatten batch and num_spans
        attention_scores = self.attention_layer(flat_span_reps).squeeze(-1)  # Shape [batch_size*num_spans, max_span_width]
        attention_weights = F.softmax(attention_scores.masked_fill(~valid_token_masks.view(-1, max_span_width), neg_limit), dim=1)
        # Compute the weighted sum
        attended_spans = (attention_weights.unsqueeze(-1) * flat_span_reps).sum(dim=1)  # Shape [batch_size*num_spans, hidden_dim]
        return attended_spans.view(batch_size, -1, hidden_dim)  # Reshape back to [batch_size, num_spans, hidden_dim]
    


    def attn_pooler_v(self, token_reps, span_starts, span_ends, span_masks, neg_limit):
        batch_size, _, seq_len, hidden_dim = token_reps.shape
        num_spans = span_starts.size(1)
        # Expand range to [batch_size, num_spans, max_span_width]
        max_span_width = (span_ends - span_starts + 1).max()
        range_tensor = torch.arange(max_span_width, device=token_reps.device).expand(batch_size, num_spans, max_span_width)
        # Create span indices [batch_size, num_spans, max_span_width]
        span_indices = span_starts.unsqueeze(-1) + range_tensor
        # Mask for valid span indices
        valid_indices = (span_indices < span_ends.unsqueeze(-1) + 1) & (span_indices < seq_len)
        valid_indices &= span_masks.unsqueeze(-1)
        # Flatten everything to align with flattened token_reps for gathering
        span_indices_flat = span_indices.masked_fill(~valid_indices, 0)  # Replace invalid indices with 0
        token_reps_expanded = token_reps.expand(batch_size, num_spans, seq_len, hidden_dim)
        # Gather tokens based on span indices
        gathered_tokens = torch.gather(token_reps_expanded, 2, span_indices_flat.unsqueeze(-1).expand(-1, -1, -1, hidden_dim))
        # Apply mask to invalidate out-of-bound tokens
        gathered_tokens *= valid_indices.unsqueeze(-1).type_as(gathered_tokens)
        # Flatten the batch and num_spans dimensions together for attention application
        gathered_tokens_flat = gathered_tokens.view(-1, max_span_width, hidden_dim)
        # Apply attention layer
        attention_scores = self.attention_layer(gathered_tokens_flat).squeeze(-1)
        attention_weights = F.softmax(attention_scores.masked_fill(~valid_indices.view(-1, max_span_width), neg_limit), dim=1)
        # Compute weighted sums
        attended_spans = (attention_weights.unsqueeze(-1) * gathered_tokens_flat).sum(dim=1)
        
        return attended_spans.view(batch_size, num_spans, hidden_dim)


    def forward(self, token_reps, span_ids, span_masks, cls_reps=None, span_widths=None, neg_limit=None, alpha=None, **kwargs):
        '''
        token_reps: (batch, seq_len, hidden)
        span_ids is of shape    (batch, num_spans, 2)       where num_spans = w_seq_len*max_span_width, values in last dim are w/sw aligned based on pooling
        span_masks is of shape   (batch, num_spans)          where num_spans = w_seq_len*max_span_width
        cls_reps: (batch, hidden)
        span_widths => the span word token widths used for width embeddings

        NOTE: for span_ids as end are python list style (actual + 1), end is always > start.
        edge cases:
        - span of width 0   => invalid span with start/end = 0, give all 0s for span rep
        - span of width 1   => internal dim = start_rep*3 + width_emb + [cls_rep]
        - span of width 2   => internal dim = start_rep*2 + end_rep  + width_emb + [cls_rep]
        - span of width > 2 => internal dim = start_rep + maxpool_inner_rep + end_rep  + width_emb + [cls_rep]
        '''
        num_spans = span_ids.shape[1]
        #extract the span_reps as start + inner_maxpool + end
        token_reps, span_starts, span_ends = extract_span_reps(token_reps, 
                                                               span_ids, 
                                                               span_masks, 
                                                               mode = 'self-attention',
                                                               neg_limit = neg_limit,
                                                               alpha = alpha)
        
        #span_reps = self.attn_pooler(token_reps, span_starts, span_ends, span_masks, neg_limit)
        span_reps = self.attn_pooler_v(token_reps, span_starts, span_ends, span_masks, neg_limit)

        # Get width embeddings
        width_emb = self.width_embeddings(span_widths)
        #Combine Components
        span_reps = torch.cat([span_reps, width_emb], dim=-1)
        
        if self.cls_flag and cls_reps is not None:
            #Expand cls_reps
            cls_expanded = cls_reps.unsqueeze(1).expand(-1, num_spans, -1)
            span_reps = torch.cat([span_reps, cls_expanded], dim=-1)

        span_reps = self.out_project(span_reps)
        return span_reps



#################################################################
#################################################################
#################################################################



###################################################################
###################################################################
###################################################################
#Utility Code
###################################################################
def extract_rep(token_reps, ids):
    '''
    This is code to support the graphER stuff, I do not use it for my stuff...
    Doesn't work for no pooling cases

    This is like a tensor friendly lookup function, it looks up the elements from a 3 dim tensor (b,s,d) 'token_reps' given a 2 dim tensor (b,k) 'ids'
    returning a tensor of shape (b,k,d), i.e. one row of d for each element k

    inputs are:
        token_reps of shape (batch, seq_len, hidden) float
        ids which are the word token ids to exract from token_reps, of shape (batch, num_spans) int, one for each span
        span_masks is the span mask indicating which spans are valid and not padding   (batch num_spans) bool
    
    The code gets the token_reps elements for the ids using gather, which is basically a lookup

    output:
        the tensor of reps for each element in ids of shape (batch, num_spans, hidden)
    '''
    hidden = token_reps.shape[2] 

    #Original gather-based implementation for word-aligned tokens
    expanded_ids = ids.unsqueeze(-1).expand(-1, -1, hidden)   #expand ids to (batch, num_spans, hidden)
    output = torch.gather(token_reps, 1, expanded_ids)
    
    return output    #shape (batch, num_spans, hidden)   where num_spans = w_seq_len*max_span_width




def extract_span_reps(token_reps, span_ids, span_masks, mode='start_end', neg_limit=None, alpha=1):
    '''
    Vectorized version of span representation extraction.
    Extracts different types of span representations based on the provided mode, using either word or subword indices.

    Args:
        token_reps: (batch, batch_max_seq_len, hidden) - Token representations, could be at the word or subword level
        span_ids: (batch, num_spans, 2) - Start and end+1 indices for spans
        span_masks: (batch, num_spans) - Boolean mask indicating valid spans
        mode: 'maxpool', 'start_end', 'start_inner_maxpool_end' - Determines the type of span representation

    NOTE: Ensure the span_masks and span_ids correctly represent the span boundaries.
    '''
    batch_max_seq_len = token_reps.shape[1]
    #get the span start and end ids
    span_starts, span_ends = span_ids[..., 0], span_ids[..., 1]  # (batch, num_spans)
    # Calculate dynamic window sizes
    span_lengths = span_ends - span_starts  # (batch, num_spans)
    ##############################
    #convert ends to actual
    #from this point span_ends are actual in this function!!!
    span_ends = span_ends - 1
    ##############################
    #determine the window
    if alpha == 1:
        # Set all window sizes to 1 if alpha is exactly 1
        win = torch.ones_like(span_lengths)
    else:
        # Calculate the window size based on alpha otherwise
        raw_win = torch.round(span_lengths.float() * alpha).long()
        # Clamp the window size to be at least 1 and at most equal to span_lengths
        win = raw_win.clamp(min=1)
        win = torch.minimum(win, span_lengths)
    
    # Prepare a range tensor
    range_tensor = torch.arange(batch_max_seq_len, device=token_reps.device).reshape(1, 1, -1)
    token_reps = token_reps.unsqueeze(1)  # (batch, 1, seq_len, hidden)


    if mode == 'self-attention':
        return token_reps, span_starts, span_ends
    
    elif mode == 'maxpool':
        #make the full mask
        #this will be a tensor of shape (batch, num_spans, sq_len) that has 1 for the tokens that are in each span
        ################################################################
        full_span_masks = (range_tensor >= span_starts.unsqueeze(-1)) & \
                          (range_tensor <= span_ends.unsqueeze(-1))  # (batch, num_spans, batch_max_seq_len)
        # Apply the provided span_masks to filter valid spans
        full_span_masks = full_span_masks & span_masks.unsqueeze(-1)  # (batch, num_spans, batch_max_seq_len)
        #Expand dimensions of the full_span_masks for compatibility with the .where command
        full_span_masks = full_span_masks.unsqueeze(-1)  # (batch, num_spans, batch_max_seq_len, 1)
        ###############################################################
        # Apply span mask and maxpool
        span_reps = torch.where(full_span_masks, token_reps, torch.full_like(token_reps, neg_limit)).max(dim=2)[0]

    else:
        start_mask = (range_tensor >= span_starts.unsqueeze(-1)) & \
                     (range_tensor <= (span_starts + win - 1).unsqueeze(-1)) & \
                      span_masks.unsqueeze(-1)
        start_mask = start_mask.unsqueeze(-1)  # (batch, num_spans, seq_len, 1)

        end_mask = (range_tensor <= span_ends.unsqueeze(-1)) & \
                   (range_tensor >= (span_ends - win + 1).unsqueeze(-1)) & \
                    span_masks.unsqueeze(-1)
        end_mask = end_mask.unsqueeze(-1)

        start_reps = torch.where(start_mask, token_reps, torch.full_like(token_reps, neg_limit)).max(dim=2)[0]
        end_reps = torch.where(end_mask, token_reps, torch.full_like(token_reps, neg_limit)).max(dim=2)[0]

        if mode == 'start_end':
            span_reps = torch.cat([start_reps, end_reps], dim=-1)

        elif mode == 'start_inner_maxpool_end':
            #get the raw inner start and end
            inner_start = span_starts + win
            inner_end = span_ends - win
            # Identify spans with inner tokens
            has_inner = (inner_end >= inner_start).unsqueeze(-1)
            # Build mask for inner tokens (inclusive)
            inner_mask = (range_tensor >= inner_start.unsqueeze(-1)) & \
                         (range_tensor <= inner_end.unsqueeze(-1)) & \
                         has_inner & span_masks.unsqueeze(-1)
            inner_mask = inner_mask.unsqueeze(-1)  # (batch, num_spans, 1)
            # Compute inner reps or reuse start reps
            inner_reps_raw = torch.where(inner_mask, token_reps, torch.full_like(token_reps, neg_limit)).max(dim=2)[0]
            # Use start_reps when no valid inner tokens
            inner_reps = torch.where(has_inner, inner_reps_raw, start_reps)
            # Final concatenation
            span_reps = torch.cat([start_reps, inner_reps, end_reps], dim=-1)

        else:
            raise ValueError(f'Invalid mode: {mode}')

    span_reps[~span_masks] = 0
    return span_reps



    

###################################################################
###################################################################
###################################################################



class SpanRepLayer(nn.Module):
    """
    Various span representation approaches

    The init call from models....
        self.span_rep_layer = SpanRepLayer(
            #specifically named....
            span_mode             = config.span_mode,
            max_seq_len           = config.max_seq_len,       #in word widths    
            max_span_width        = config.max_span_width,    #in word widths
            #the rest are in kwargs...
            hidden_size           = config.hidden_size,
            width_embeddings      = self.width_embeddings,    #in word widths
            dropout               = config.dropout,
            layer_type            = config.projection_layer_type,
            ffn_ratio             = config.ffn_ratio, 
            cls_flag              = config.model_source == 'HF'    #whether we will have a cls token rep
        )

    """
    def __init__(self, span_mode, max_span_width, **kwargs):
        super().__init__()
        #kwargs has remaining: ['hidden_size', 'width_embeddings', 'dropout', layer_type, 'ffn_ratio', 'use_span_pos_encoding', 'cls_flag']
        self.span_mode = span_mode

        if span_mode == 'firstlast_grapher':
            self.span_rep_layer = First_n_Last_graphER(max_span_width, **kwargs)
        elif span_mode == 'firstlast':
            self.span_rep_layer = First_n_Last(max_span_width, **kwargs)
        elif span_mode == 'spert':
            self.span_rep_layer = Spert(max_span_width, **kwargs)
        elif span_mode == 'nathan':
            self.span_rep_layer = Nathan(max_span_width, **kwargs)
        elif span_mode == 'attn':
            self.span_rep_layer = Attn(max_span_width, **kwargs)
        else:
            raise ValueError(f'Unknown span mode {span_mode}')



    def forward(self, token_reps, span_ids, span_masks, **kwargs):
        span_reps = self.span_rep_layer(token_reps, span_ids, span_masks, **kwargs)
        #apply the span_mask to neutralise the unwanted span_reps, not really needed but just for safety
        span_reps = span_reps * span_masks.unsqueeze(-1)
        return span_reps

