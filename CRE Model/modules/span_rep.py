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

    def forward(self, token_reps, span_ids, span_masks, pooling, **kwargs):
        '''
        token_reps is of shape  (batch, seq_len, hidden)    where seq_len is w_seq_len for word token aligned token_reps
        span_ids is of shape    (batch, num_spans, 2)       where num_spans = w_seq_len*max_span_width, values in last dim are w/sw aligned based on pooling
        span_masks is of shape   (batch, num_spans)          where num_spans = w_seq_len*max_span_width
        Because when they generate all possible spans they just generate all spans from width 1 to max_span_width for each start idx, even when the end idx is out of the max_seq_len, they just mask those as invalid spans 
        So that view statement at the end will work ok, it just kind of associates the max_span_width spans with each start token idx in a way
        '''
        if not pooling:
            raise Exception('graphER span generation only works for the pooled cases')

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


    def forward(self, token_reps, span_ids, span_masks, pooling, neg_limit=None, **kwargs):
        '''
        token_reps is of shape  (batch, seq_len, hidden)    => w_seq_len for word token aligned, or sw_seq_len for sw token aligned, it matters!!
        span_ids is of shape    (batch, num_spans, 2)       where num_spans = w_seq_len*max_span_width, values in last dim are w/sw aligned based on pooling
        span_masks is of shape   (batch, num_spans)          where num_spans = w_seq_len*max_span_width
        pooling is true if we are word token aligned, false if we are sw token aligned
        '''
        #extract the start and end span reps
        span_reps = extract_span_reps(token_reps, 
                                      span_ids, 
                                      span_masks, 
                                      pooling, 
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


    def forward(self, token_reps, span_ids, span_masks, pooling, cls_reps=None, span_widths=None, neg_limit=None, **kwargs):
        '''
        token_reps: (batch, seq_len, hidden)
        span_ids is of shape    (batch, num_spans, 2)       where num_spans = w_seq_len*max_span_width, values in last dim are w/sw aligned based on pooling
        span_masks is of shape   (batch, num_spans)          where num_spans = w_seq_len*max_span_width
        pooling => true if w token aligned
        cls_reps: (batch, hidden)
        span_widths => the span word token widths used for width embeddings
        '''
        num_spans = span_ids.shape[1]
        # Get maxpooled span representations
        span_maxpool_reps = extract_span_reps(token_reps, 
                                              span_ids, 
                                              span_masks, 
                                              pooling, 
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

class Nathan_v1(nn.Module):
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


    def forward(self, token_reps, span_ids, span_masks, pooling, cls_reps=None, span_widths=None, neg_limit=None, **kwargs):
        '''
        token_reps: (batch, seq_len, hidden)
        span_ids is of shape    (batch, num_spans, 2)       where num_spans = w_seq_len*max_span_width, values in last dim are w/sw aligned based on pooling
        span_masks is of shape   (batch, num_spans)          where num_spans = w_seq_len*max_span_width
        pooling => true if w token aligned
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
                                      pooling, 
                                      mode = 'start_inner_maxpool_end', 
                                      neg_limit = neg_limit)
        # Get width embeddings
        width_emb = self.width_embeddings(span_widths)
        #Combine Components
        span_reps = torch.cat([span_reps, width_emb], dim=-1)
        if self.cls_flag and cls_reps is not None:
            #Expand cls_reps
            cls_expanded = cls_reps.unsqueeze(1).expand(-1, num_spans, -1)
            span_reps = torch.cat([cls_expanded, span_reps], dim=-1)

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




def extract_span_reps(token_reps, span_ids, span_masks, pooling, mode='start_end', neg_limit=None):
    '''
    Vectorized version of span representation extraction.
    Extracts different types of span representations based on the provided mode, using either word or subword indices.

    Args:
        token_reps: (batch, batch_max_seq_len, hidden) - Token representations, could be at the word or subword level
        span_ids: (batch, num_spans, 2) - Start and end+1 indices for spans
        span_masks: (batch, num_spans) - Boolean mask indicating valid spans
        pooling: True for word token aligned reps
        mode: 'maxpool', 'start_end', 'start_inner_maxpool_end' - Determines the type of span representation

    NOTE: Ensure the span_masks and span_ids correctly represent the span boundaries.
    '''
    batch_max_seq_len = token_reps.shape[1]
    window = 1 if pooling else 3

    # Extract start and end indices from span_ids
    span_starts, span_ends = span_ids[..., 0], span_ids[..., 1]  # (batch, num_spans)
    # Generate span masks of shape (batch, num_spans, batch_max_seq_len)
    range_tensor = torch.arange(batch_max_seq_len, device=token_reps.device).reshape(1, 1, -1)
    #Expand dimensions of token_reps for compatibility with the .where commands with the masks
    token_reps = token_reps.unsqueeze(1)  # (batch, 1, batch_max_seq_len, hidden)

    #process based on mode
    if mode == 'maxpool':
        #make the full mask
        #this will be a tensor of shape (batch, num_spans, sq_len) that has 1 for the tokens that are in each span
        ################################################################
        full_span_masks = (range_tensor >= span_starts.unsqueeze(-1)) & \
                         (range_tensor < span_ends.unsqueeze(-1))  # (batch, num_spans, batch_max_seq_len)
        # Apply the provided span_masks to filter valid spans
        full_span_masks = full_span_masks & span_masks.unsqueeze(-1)  # (batch, num_spans, batch_max_seq_len)
        #Expand dimensions of the full_span_masks for compatibility with the .where command
        full_span_masks = full_span_masks.unsqueeze(-1)  # (batch, num_spans, batch_max_seq_len, 1)
        ###############################################################
        # Apply span mask and maxpool
        span_reps = torch.where(full_span_masks, token_reps, torch.full_like(token_reps, neg_limit)).max(dim=2)[0]

    else:    #for the other modes
        #make the start/end masks
        #this will be a tensor of shape (batch, num_spans, sq_len) that has 1 for the tokens that are in teh start/end window for each span
        ################################################################
        start_mask = (range_tensor >= span_starts.unsqueeze(-1)) & \
                     (range_tensor < torch.min(span_starts.unsqueeze(-1) + window, span_ends.unsqueeze(-1)))
        end_mask = (range_tensor >= torch.max(span_ends.unsqueeze(-1) - window, span_starts.unsqueeze(-1))) & \
                   (range_tensor < span_ends.unsqueeze(-1))
        # Apply the provided span_masks to filter valid spans
        start_mask = start_mask & span_masks.unsqueeze(-1)
        end_mask = end_mask & span_masks.unsqueeze(-1)
        # Expand dimensions for compatibility with .where
        start_mask = start_mask.unsqueeze(-1)  # (batch, num_spans, batch_max_seq_len, 1)
        end_mask = end_mask.unsqueeze(-1)  # (batch, num_spans, batch_max_seq_len, 1)
        ################################################################
        # Apply masks and maxpool
        start_reps = torch.where(start_mask, token_reps, torch.full_like(token_reps, neg_limit)).max(dim=2)[0]
        end_reps = torch.where(end_mask, token_reps, torch.full_like(token_reps, neg_limit)).max(dim=2)[0]

        if mode == 'start_end':
            span_reps = torch.cat([start_reps, end_reps], dim=-1)

        elif mode == 'start_inner_maxpool_end':
            #make the inner token masks
            #this will be a tensor of shape (batch, num_spans, sq_len) that has 1 for the tokens that are in the span but not including the start/end window tokens for each span
            ################################################################
            inner_mask = (range_tensor >= torch.min(span_starts.unsqueeze(-1) + window, span_ends.unsqueeze(-1))) & \
                         (range_tensor < torch.max(span_ends.unsqueeze(-1) - window, span_starts.unsqueeze(-1))) & \
                         ((span_ends - span_starts) > 2 * window).unsqueeze(-1)
            #Apply the provided span_masks to filter valid spans
            inner_mask = inner_mask & span_masks.unsqueeze(-1)
            #Expand dimensions for compatibility with .where
            inner_mask = inner_mask.unsqueeze(-1)  # (batch, num_spans, batch_max_seq_len, 1)
            ################################################################
            #Apply masks and maxpool
            inner_reps = torch.where(inner_mask, token_reps, torch.full_like(token_reps, neg_limit)).max(dim=2)[0]
            #Check if the inner_reps are == neg_limit for each span (indicating no valid inner tokens), replace with start_reps if true
            no_inner_tokens = (inner_reps == neg_limit).all(dim=-1)
            inner_reps[no_inner_tokens] = start_reps[no_inner_tokens]
            #make the final reps
            span_reps = torch.cat([start_reps, inner_reps, end_reps], dim=-1)

        else:
            raise ValueError(f'Invalid mode: {mode}')

    # Reset representations for invalid spans to zero
    span_reps[~span_masks] = 0

    return span_reps  # (batch, num_spans, h), where h is hidden, 2*hidden, or 3*hidden based on mode

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
            pooling               = config.subtoken_pooling,     #whether we are using pooling or not
            #the rest are in kwargs...
            hidden_size           = config.hidden_size,
            width_embeddings      = self.width_embeddings,    #in word widths
            dropout               = config.dropout,
            layer_type            = config.projection_layer_type,
            ffn_ratio             = config.ffn_ratio, 
            cls_flag              = config.model_source == 'HF'    #whether we will have a cls token rep
        )

    """
    def __init__(self, span_mode, max_seq_len, max_span_width, pooling, **kwargs):
        super().__init__()
        #kwargs has remaining: ['hidden_size', 'width_embeddings', 'dropout', layer_type, 'ffn_ratio', 'use_span_pos_encoding', 'cls_flag']
        self.span_mode = span_mode
        self.pooling = pooling

        if span_mode == 'firstlast_grapher':
            self.span_rep_layer = First_n_Last_graphER(max_span_width, **kwargs)
        elif span_mode == 'firstlast':
            self.span_rep_layer = First_n_Last(max_span_width, **kwargs)
        elif span_mode == 'spert':
            self.span_rep_layer = Spert(max_span_width, **kwargs)
        elif span_mode == 'nathan':
            self.span_rep_layer = Nathan_v1(max_span_width, **kwargs)
        else:
            raise ValueError(f'Unknown span mode {span_mode}')



    def forward(self, token_reps, span_ids, span_masks, **kwargs):
        #span_ids = span_ids * span_masks.unsqueeze(-1)
        span_reps = self.span_rep_layer(token_reps, span_ids, span_masks, self.pooling, **kwargs)
        #apply the span_mask to neutralise the unwanted span_reps, not really needed but just for safety
        span_reps = span_reps * span_masks.unsqueeze(-1)
        return span_reps

