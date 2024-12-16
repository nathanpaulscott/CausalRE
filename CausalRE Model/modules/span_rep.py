import math, random, os
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import torch.nn.init as init

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
    
    NOTE: span_ids with start, end = 0,0 are processed nromally, but the reps will be ignored later by the span_mask
   '''
    def __init__(self, 
                 max_span_width, 
                 hidden_size, 
                 ffn_ratio,
                 dropout, 
                 **kwargs):
        super().__init__()
        #overwrite the passed values to copy graphER
        ffn_ratio = 1.5
        dropout = 0.4
        self.max_span_width = max_span_width     #max span width
        self.project_start = FFNProjectionLayer(hidden_size, ffn_ratio, hidden_size, dropout)     #FFN for the start token
        self.project_end = FFNProjectionLayer(hidden_size, ffn_ratio, hidden_size, dropout)       #FFN for th eend token
        self.out_project = FFNProjectionLayer(2*hidden_size, ffn_ratio, hidden_size, dropout)    #projects the concat of the start/end back to hidden
        self.relu = nn.ReLU()

        self.init_weights()


    def init_weights(self):
        pass


    def forward(self, token_reps, span_ids, pooling, **kwargs):
        '''
        token_reps is of shape  (batch, seq_len, hidden)    where seq_len is w_seq_len for word token aligned token_reps
        span_ids is of shape    (batch, num_spans, 2) where num_spans = w_seq_len*max_span_width, w or sw alignment follows pooling
        Because when they generate all possible spans they just generate all spans from width 1 to max_span_width for each start idx, even when the end idx is out of the max_seq_len, they just mask those as invalid spans 
        So that view statement at the end will work ok, it just kind of associates the max_span_width spans with each start token idx in a way
        '''
        if not pooling:
            raise Exception('graphER span generation only works for the pooled cases')
        #run the token_reps through a simple FFN for the start and end token
        #it actually does nothing, but further mix the token embeddings around
        #is this really neccessary?
        start_rep = self.project_start(token_reps)
        end_rep = self.project_end(token_reps)
        #extract the start rep and end rep out of token_reps
        start_span_rep = extract_rep(start_rep, span_ids[:, :, 0])
        end_span_rep = extract_rep(end_rep, span_ids[:, :, 1] - 1)
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

    NOTE: span_ids with start, end = 0,0 are processed normally, but the reps will be ignored later by the span_mask
   '''
    def __init__(self, max_span_width, hidden_size, **kwargs):
        super().__init__()
        self.out_project = nn.Linear(2*hidden_size, hidden_size)    #projects the concat of the start/end back to hidden

        self.init_weights()


    def init_weights(self):
        init.xavier_normal_(self.out_project.weight)
        init.constant_(self.out_project.bias, 0)


    def forward(self, token_reps, span_ids, pooling, **kwargs):
        '''
        token_reps is of shape  (batch, seq_len, hidden)    => w_seq_len for word token aligned, or sw_seq_len for sw token aligned, it matters!!
        span_ids is of shape    (batch, num_spans, 2) where num_spans = w_seq_len*max_span_width, w or sw alignment follows pooling
        pooling is true if we are word token aligned, false if we are sw token aligned
        '''
        #extract the start and end span reps
        span_reps = extract_span_reps(token_reps, span_ids, pooling, mode='start_end')     
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

    NOTE: span_ids with start, end = 0,0 are processed nromally, but the reps will be ignored later by the span_mask
    '''    
    def __init__(self, 
                 max_span_width, 
                 hidden_size, 
                 width_embeddings, 
                 cls_flag,
                 **kwargs):
        super().__init__()
        #kwargs has remaining: ['dropout', 'ffn_ratio', 'use_span_pos_encoding']
        self.max_span_width = max_span_width
        self.width_embeddings = width_embeddings
        internal_hidden_size = hidden_size + width_embeddings.embedding_dim
        self.cls_flag = cls_flag
        if self.cls_flag:
            internal_hidden_size += hidden_size

        self.out_project = nn.Linear(internal_hidden_size, hidden_size)

        self.init_weights()


    def init_weights(self):
        init.xavier_normal_(self.out_project.weight)
        init.constant_(self.out_project.bias, 0)


    def forward(self, token_reps, span_ids, pooling, cls_reps=None, span_widths=None, **kwargs):
        '''
        token_reps: (batch, seq_len, hidden)
        span_ids: (batch, num_spans, 2)     #word/sw token aligned depends on pooling
        w_span_ids: (batch, num_spans, 2)     #word token aligned
        pooling => true if w token aligned
        cls_reps: (batch, hidden)
        span_widths => the span word token widths used for width embeddings
        '''
        batch, num_spans, _ = span_ids.shape
        # Get maxpooled span representations
        #NOTE: always takes w_span_ids here, jsut send the w2sw_map if we have no pooling
        span_maxpool_reps = extract_span_reps(token_reps, span_ids, pooling, mode='maxpool')
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

    NOTE: span_ids with start, end = 0,0 are processed normally, but the reps will be ignored later by the span_mask
    '''    
    def __init__(self, 
                 max_span_width, 
                 hidden_size, 
                 width_embeddings, 
                 cls_flag,
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

        self.out_project = nn.Linear(internal_hidden_size, hidden_size)

        self.init_weights()


    def init_weights(self):
        init.xavier_normal_(self.out_project.weight)
        init.constant_(self.out_project.bias, 0)


    def forward(self, token_reps, span_ids, pooling, cls_reps=None, span_widths=None, **kwargs):
        '''
        token_reps: (batch, seq_len, hidden)
        span_ids: (batch, num_spans, 2)     #word/sw token aligned depends on pooling
        w_span_ids: (batch, num_spans, 2)     #word token aligned
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
        batch, num_spans, _ = span_ids.shape
        #extract the span_reps as start + inner_maxpool + end
        span_reps = extract_span_reps(token_reps, span_ids, pooling, mode='start_inner_maxpool_end')
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


class AttentionPooling(nn.Module):
    '''
    This uses span_ids, max_seq_len and max_span_width which have been selected depending on whether we are using pooling or not
    for pooling => all are word token aligned
    for no pooling => all are sw token aligned     
    '''
    def __init__(self, 
                 max_seq_len,      #in word tokens for pooling sw tokens for no pooling
                 max_span_width,  #in word tokens for pooling sw tokens for no pooling
                 hidden_size, 
                 ffn_ratio=4, 
                 num_heads=4, 
                 dropout=0.1, 
                 use_span_pos_encoding=False,
                 **kwargs):
        super().__init__()
        self.hidden_size = hidden_size
        self.use_span_pos_encoding = use_span_pos_encoding
        #attention pooling block: dummy query attends to span tokens
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(hidden_size)
        self.ffn = FFNProjectionLayer(hidden_size, ffn_ratio=ffn_ratio, dropout=dropout)
        # Learnable query vector for attention pooling block
        self.dummy_query = nn.Parameter(torch.randn(1, 1, hidden_size))
        # Positional encodings for full sequence and span-internal positions
        self.seq_pos_encoder = PositionalEncoding(hidden_size, dropout, max_seq_len)
        if use_span_pos_encoding:
            self.span_pos_encoder = PositionalEncoding(hidden_size, dropout, max_span_width)
        
        self.init_weights()


    def init_weights(self):
        # Init dummy query
        init.xavier_normal_(self.dummy_query)


    def forward(self, token_reps, span_ids, pooling, **kwargs):
        '''
        Args:
            token_reps: (batch, max_batch_seq_len, hidden)    #sw or word token aligned
            span_ids: (batch, num_spans, 2)                   #same alignment as token_reps
            pooling if word token alinged or not
            
        Returns:
            span_reps: (batch, num_spans, hidden)

        token_reps are either w token aligned or sw_token aligned
        - if they are sw_token aligned, the span_ids passed are also sw_token aligned and masked
        - if they are word token aligned, the span_ids passed are also word token aligned and masked
        '''
        batch, seq_len, hidden = token_reps.shape
        batch, num_spans, _ = span_ids.shape
        
        # Add positional encodings to full sequence
        token_reps = self.seq_pos_encoder(token_reps)
        
        all_span_reps = []
        print(self.dummy_query)
        for obs_id in range(batch):
            span_reps = []
            for span_id in range(num_spans):
                start, end = span_ids[obs_id, span_id]
                # Handle invalid spans
                if end - start == 0:
                    span_reps.append(torch.zeros(hidden, device=token_reps.device))
                    continue
                # Extract span tokens (already batch first)
                span_token_reps = token_reps[obs_id, start:end].unsqueeze(0)   # (1, span_len, hidden)
                # Add span-internal positional encodings if enabled
                if self.use_span_pos_encoding:
                    span_token_reps = self.span_pos_encoder(span_token_reps)
                # Attention block: dummy query attends to span tokens
                # Keep batch_first format
                dummy_query = self.dummy_query  # (1, 1, hidden)
                output, _ = self.attn(dummy_query, span_token_reps, span_token_reps, need_weights=False)
                # Already batch first for norm and ffn
                output = self.norm(output + dummy_query)
                output = self.norm(self.ffn(output) + output)
                span_reps.append(output[0, 0])  # Extract final vector

            all_span_reps.append(torch.stack(span_reps))
            
        output = torch.stack(all_span_reps)
        return output   # (batch, num_spans, hidden)
    




class AttentionPooling_vectorized(nn.Module):
    '''
    I have tried to verify why this gives differetn results to the non-vectorised version and am not able,
    it may be ok, may be a problem, I suggest you need to write your own mha code for verification of what is going on
    to do later, I tried the key padding mask as boolean and float with -inf, got the same results, which are different to the loop case
    '''
    def __init__(self, 
                max_seq_len,      #in word tokens for pooling sw tokens for no pooling
                max_span_width,  #in word tokens for pooling sw tokens for no pooling
                hidden_size, 
                ffn_ratio=4, 
                num_heads=4, 
                dropout=0.1, 
                use_span_pos_encoding=False,
                **kwargs):
        super().__init__()
        self.hidden_size = hidden_size
        self.use_span_pos_encoding = use_span_pos_encoding

        # Attention pooling block: dummy query attends to span tokens
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(hidden_size)
        self.ffn = FFNProjectionLayer(hidden_size, ffn_ratio=ffn_ratio, dropout=dropout)
        # Learnable query vector for attention pooling block
        self.dummy_query = nn.Parameter(torch.randn(1, 1, hidden_size))
        # Positional encodings for full sequence and span-internal positions
        self.seq_pos_encoder = PositionalEncoding(hidden_size, dropout, max_seq_len)
        if use_span_pos_encoding:
            self.span_pos_encoder = PositionalEncoding(hidden_size, dropout, max_span_width)
        
        self.init_weights()

    def init_weights(self):
        # Init dummy query
        init.xavier_normal_(self.dummy_query)


    def forward(self, token_reps, span_ids, pooling, **kwargs):
        '''
        Args:
            token_reps: (batch, max_batch_seq_len, hidden)    #sw or word token aligned
            span_ids: (batch, num_spans, 2)                   #same alignment as token_reps
            pooling if word token alinged or not
            
        Returns:
            span_reps: (batch, num_spans, hidden)

        token_reps are either w token aligned or sw_token aligned
        - if they are sw_token aligned, the span_ids passed are also sw_token aligned and masked
        - if they are word token aligned, the span_ids passed are also word token aligned and masked

        NOTE: I have checked this and it gives different results to the older loop style code, but the code is good
        without re-writing the mha code myself and getting into it, I just do not know, but the way it is done here is ok
        '''
        batch_size, seq_len, hidden = token_reps.shape
        batch_size, num_spans, _ = span_ids.shape
        # Add positional encodings to full sequence
        token_reps = self.seq_pos_encoder(token_reps)

        # Get maximum span length
        span_lengths = span_ids[:, :, 1] - span_ids[:, :, 0]  # (batch, num_spans)
        max_span_len = max(span_lengths.max().item(), 1)

        # Initialize tensors for batched processing
        span_token_reps = torch.zeros((batch_size, num_spans, max_span_len, hidden), device=token_reps.device)
        key_padding_mask = torch.ones((batch_size, num_spans, max_span_len), device=token_reps.device, dtype=torch.bool)  # True means ignore position
        #key_padding_mask = torch.full((batch_size, num_spans, max_span_len), float('-inf'), device=token_reps.device)  # Use -inf instead of True
        
        # Extract span tokens and create attention masks
        for b in range(batch_size):
            for s in range(num_spans):
                start, end = span_ids[b, s]
                span_len = end - start
                if span_len > 0:
                    # Copy span tokens
                    span_token_reps[b, s, :span_len, :] = token_reps[b, start:end, :]
                    # Update attention mask for valid tokens
                    key_padding_mask[b, s, :span_len] = False
                    #key_padding_mask[b, s, :span_len] = 0  # Set to 0 where attention is needed

        # Add span positional encodings if enabled
        if self.use_span_pos_encoding:
            span_token_reps = self.span_pos_encoder(span_token_reps)
        
        '''
        #print(key_padding_mask)
        print(self.dummy_query)
        
        for b in range(span_token_reps.shape[0]):
            for s in range(span_token_reps.shape[1]):
                print(span_token_reps[b,s])
                print(key_padding_mask[b,s])
        '''

        # Reshape tensors for batched attention - keeping each span separate
        # Each dummy-span pair becomes a separate "batch" item
        dummy_query = self.dummy_query.expand(batch_size * num_spans, 1, hidden)
        span_token_reps = span_token_reps.view(batch_size * num_spans, max_span_len, hidden)
        key_padding_mask = key_padding_mask.view(batch_size * num_spans, max_span_len)

        # Apply attention with masking
        output, _ = self.attn(
            dummy_query,
            span_token_reps,
            span_token_reps,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        # Apply layer norm and FFN, can do it here or after reshaping, doesn't matter
        output = self.norm(output + dummy_query)
        output = self.norm(self.ffn(output) + output)

        # Reshape back to batch dimensions
        output = output.view(batch_size, num_spans, hidden)  # (batch, num_spans, hidden)
        
        # Zero out empty spans
        empty_spans = (span_lengths == 0)
        output[empty_spans] = 0
        
        return output  # (batch, num_spans, hidden)



#############################################################
#Utility Code################################################
#############################################################
class FFNProjectionLayer(nn.Module):
    '''
    Projection Layer that can have nonlinearities, dropout and FFN aspects
    '''
    def __init__(self, input_dim, ffn_ratio=1.5, output_dim=None, dropout=0.2):
        super().__init__()
        #check if output dim is specified, if not then FFN outputs same dimensionality as the input
        if output_dim is None: 
            output_dim = input_dim
        intermed_dim = int(output_dim * ffn_ratio)

        self.linear_in = nn.Linear(input_dim, intermed_dim)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.linear_out = nn.Linear(intermed_dim, output_dim)
        
        self.init_weights()

    def init_weights(self):
        init.kaiming_normal_(self.linear_in.weight, nonlinearity='relu')
        init.constant_(self.linear_in.bias, 0)
        init.kaiming_normal_(self.linear_out.weight, nonlinearity='relu')
        init.constant_(self.linear_out.bias, 0)

    def forward(self, x):
        x = self.linear_in(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.linear_out(x)
        x = self.dropout(x)
        return x



class PositionalEncoding(nn.Module):
    def __init__(self, hidden_size, dropout=0.1, max_seq_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_size, 2) * (-math.log(10000.0) / hidden_size))
        pe = torch.zeros(1, max_seq_len, hidden_size)  # batch first format
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        """
        Args:
            x: Tensor shape (batch, seq_len, hidden)
        Returns:
            Same shape as input
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)    


def extract_rep(token_reps, w_span_ids, w2sw_map=None):
    '''
    This is code to support the graphER stuff, I do not use it for my stuff...
    Doesn't work for no pooling cases

    This is like a tensor friendly lookup function, it looks up the elements from a 3 dim tensor (b,s,d) 'token_reps' given a 2 dim tensor (b,k) 'ids'
    returning a tensor of shape (b,k,d), i.e. one row of d for each element k

    inputs are:
        token_reps of shape (batch, seq_len, hidden)
        ids which are the ids to exract from token_reps, of shape (batch, num_spans)
    
    The code gets the token_reps elements for the ids using gather, which is basically a lookup

    output:
        the tensor of reps for each element in ids of shape (batch, num_spans, hidden)

    NOTE: if w2sw_map is present, then the token_reps is sw token aligned, so need to use w2sw_map to get the sw token ids for each w token id and max pool them
    '''
    batch, seq_len, hidden = token_reps.shape       #seq_len is sw_seq_len for sw aligned case or w_seq_len for w token aligned case
    batch, num_spans = w_span_ids.shape

    # Original gather-based implementation for word-aligned tokens
    expanded_ids = w_span_ids.unsqueeze(-1).expand(-1, -1, hidden)
    output = torch.gather(token_reps, 1, expanded_ids)
    
    return output    #shape (batch, num_spans, hidden)   where num_spans = w_seq_len*max_span_width



def extract_span_reps(token_reps, span_ids, pooling, mode='start_end'):
    '''
    Extracts different types of span representations based on the provided mode, using either word or subword indices.

    Args:
        token_reps: (batch, seq_len, hidden) - Token representations, could be at the word or subword level
        span_ids: (batch, num_spans, 2) - Word or sw aligned span indices dependent on pooling being true/false
        pooling: true for w token aligned reps
        mode: 'maxpool', 'start_end', 'start_inner_maxpool_end' - Determines the type of span representation

    Returns:
        span_reps: Tensor with appropriate dimensions based on the selected mode
    '''
    batch, _, hidden = token_reps.shape
    all_span_reps = []
    #set the subtoken first and last windows (using this as a workaround to not having w2sw_map to get the actual first and end subtoken widths)
    #this many subword tokens are used for the first/last part of the span, if the span is shorter than this,then the window is the whole span
    #if the start window and end window overlap as the span length in sw tokens is < 2*sw_win, then the inner subtokens are None
    win = 1 if pooling else 3   
    
    for obs_id in range(batch):
        obs_spans = []
        for span_id in range(len(span_ids[obs_id])):
            start, end = span_ids[obs_id, span_id]
            span_len = max(0, end - start)

            #deal with invalid span cases
            if span_len == 0:
                if mode == 'maxpool': rep=1
                elif mode == 'start_end': rep=2
                elif mode == 'start_inner_maxpool_end': rep=3
                else: raise Exception(f'error: span_mode is incorrect: {mode}')
                obs_spans.append(torch.zeros(rep*hidden, device=token_reps.device))
                continue                

            #do maxpooling mode, this is the easiest
            #this works for both the pooling or no pooling cases
            if mode == 'maxpool':
                maxpool_rep = torch.max(token_reps[obs_id, start:end], dim=0)[0]
                obs_spans.append(maxpool_rep)
                continue
            
            #at this point we are in either start_inner_maxpool_end or start_end mode
            #this works for both the pooling or no pooling cases
            #get the start and end reps first
            start_start, end_end = start, end
            start_end = min(start_start + win, end_end)  # Ensure not exceeding end
            end_start = max(end_end - win, start_start)  # Ensure not preceding start
            #NOTE: in the case that the span_len <= win, then start_rep = end_rep
            start_rep = torch.max(token_reps[obs_id, start_start:start_end], dim=0)[0]
            end_rep = torch.max(token_reps[obs_id, end_start:end_end], dim=0)[0]

            if mode == 'start_end':
                obs_spans.append(torch.cat([start_rep, end_rep], dim=-1))
                continue
            
            #if we got here we are in start_inner_maxpool_end mode
            if span_len > 2*win:   #the case we have some inner tokens in the span
                inner_rep = torch.max(token_reps[obs_id, start+win:end-win], dim=0)[0]
            else:    #the case the span is too short and we have no inner tokens, so just use the start rep for the inner rep
                inner_rep = start_rep
            obs_spans.append(torch.cat([start_rep, inner_rep, end_rep], dim=-1))

        all_span_reps.append(torch.stack(obs_spans))

    return torch.stack(all_span_reps)





def extract_span_reps_vectorized(token_reps, span_ids, pooling, mode='start_end'):
    '''
    Vectorized version of span representation extraction.
    Extracts different types of span representations based on the provided mode, using either word or subword indices.

    Args:
        token_reps: (batch, seq_len, hidden) - Token representations, could be at the word or subword level
        span_ids: (batch, num_spans, 2) - Word or sw aligned span indices dependent on pooling being true/false
        pooling: true for w token aligned reps
        mode: 'maxpool', 'start_end', 'start_inner_maxpool_end' - Determines the type of span representation

    NOTE: watch out for gpt and claude, they will try to mask then maxpool, which is a common gotcha, if you mask then maxpool, you have to convert to -inf for the masked out values        
    '''
    batch, seq_len, hidden = token_reps.shape
    #set the subtoken first and last windows (using this as a workaround to not having w2sw_map to get the actual first and end subtoken widths)
    #this many subword tokens are used for the first/last part of the span, if the span is shorter than this,then the window is the whole span
    #if the start window and end window overlap as the span length in sw tokens is < 2*sw_win, then the inner subtokens are None
    win = 1 if pooling else 3   

    # Calculate span lengths and create a mask for valid spans
    span_starts, span_ends = span_ids[..., 0], span_ids[..., 1]    # int (batch, num_spans)
    valid_spans_mask = span_ends > span_starts  # Ensuring end is greater than start    #bool (batch, num_spans)

    # Precompute span masks for maxpool and boundary conditions for other modes
    range_tensor = torch.arange(seq_len, device=token_reps.device).reshape(1, 1, -1)  #int (1, 1, seq_len) => range tensor is a vector from 0 to seq_len-1
    span_mask = (range_tensor >= span_starts.unsqueeze(-1)) & \
                (range_tensor < span_ends.unsqueeze(-1))   #bool (batch, num_spans, seq_len) => for each span in each batch we have a bool vector with zero if the token is out of the span and 1 if it is in the span
    #unsqueeze these 2 tensors so they are compatible
    span_mask = span_mask.unsqueeze(-1)
    token_reps = token_reps.unsqueeze(1)
    if mode == 'maxpool':
        #Apply span mask using torch.where and apply maxpooling
        #this handles expansion of all dims so both tensor match the largest of each pair of dims
        #note how it converts the mask False to -inf as we will maxpool the output
        span_reps = torch.where(span_mask, token_reps, torch.full_like(token_reps, float('-inf'))).max(dim=2)[0]

    else:
        # Compute start and end masks using torch.where for proper handling of masked areas
        start_mask = (range_tensor >= span_starts.unsqueeze(-1)) & \
                     (range_tensor < torch.min(span_starts.unsqueeze(-1) + win, span_ends.unsqueeze(-1)))     #bool (batch, num_spans, seq_len) => for each span in each batch we have a bool vector with zero if the token is out of the span start win
        end_mask = (range_tensor >= torch.max(span_ends.unsqueeze(-1) - win, span_starts.unsqueeze(-1))) & \
                   (range_tensor < span_ends.unsqueeze(-1))       #bool (batch, num_spans, seq_len) => for each span in each batch we have a bool vector with zero if the token is out of the span end win
        #unsqueeze these 2 tensors so they are compatible with token reps
        start_mask = start_mask.unsqueeze(-1)
        end_mask = end_mask.unsqueeze(-1)
        #Apply masks with where and apply maxpooling
        #note how it converts the mask False to -inf as we will maxpool the output
        start_reps = torch.where(start_mask, token_reps, torch.full_like(token_reps, float('-inf'))).max(dim=2)[0]
        end_reps = torch.where(end_mask, token_reps, torch.full_like(token_reps, float('-inf'))).max(dim=2)[0]

        if mode == 'start_end':
            span_reps = torch.cat([start_reps, end_reps], dim=-1)

        elif mode == 'start_inner_maxpool_end':
            # Compute mask for inner tokens if span length allows
            inner_mask = (range_tensor >= torch.min(span_starts.unsqueeze(-1) + win, span_ends.unsqueeze(-1))) & \
                         (range_tensor < torch.max(span_ends.unsqueeze(-1) - win, span_starts.unsqueeze(-1))) & \
                         ((span_ends - span_starts) > 2 * win).unsqueeze(-1)
            inner_mask = inner_mask.unsqueeze(-1)
            inner_reps = torch.where(inner_mask, token_reps, torch.full_like(token_reps, float('-inf'))).max(dim=2)[0]
            # Check if the inner_reps are -inf for each span (indicating no valid inner tokens), if true, replace with start_reps
            no_inner_tokens = torch.isinf(inner_reps).all(dim=-1)  # Check along the hidden dimension
            inner_reps[no_inner_tokens] = start_reps[no_inner_tokens]

            span_reps = torch.cat([start_reps, inner_reps, end_reps], dim=-1)

        else:
            raise ValueError(f'Invalid mode: {mode}')

    # Reset representations for invalid spans to zero, works regardless of hidden length
    span_reps[~valid_spans_mask] = 0

    return span_reps    #(batch, num_spans, h) => where h is hidden for maxpool, 2*hidden for start_end, 3*hidden for start_inner_maxpool_end



class SpanRepLayer(nn.Module):
    """
    Various span representation approaches

        self.span_rep_layer = SpanRepLayer(
            span_mode           = config.span_mode,
            hidden_size         = config.hidden_size,
            max_span_width      = config.max_span_width,    #in word widths
            max_seq_len         = config.max_seq_len,       #in word widths    
            width_embeddings    = self.width_embeddings,    #in word widths
            dropout             = config.dropout,
            ffn_ratio           = config.ffn_ratio, 
            use_span_pos_encoding=config.use_span_pos_encoding,    #whether to use span pos encoding in addition to full seq pos encoding
            pooling             = config.subtoken_pooling     #whether we are using pooling or not
        )



    """
    def __init__(self, span_mode, max_seq_len, max_span_width, pooling, **kwargs):
        super().__init__()
        #kwargs has remaining: ['hidden_size', 'width_embeddings', 'dropout', 'ffn_ratio', 'use_span_pos_encoding']
        self.span_mode = span_mode
        self. pooling = pooling

        if span_mode == 'firstlast_grapher':
            self.span_rep_layer = First_n_Last_graphER(max_span_width, **kwargs)
        elif span_mode == 'firstlast':
            self.span_rep_layer = First_n_Last(max_span_width, **kwargs)
        elif span_mode == 'spert':
            self.span_rep_layer = Spert(max_span_width, **kwargs)
        elif span_mode == 'nathan':
            self.span_rep_layer = Nathan_v1(max_span_width, **kwargs)
        elif span_mode == 'attentionpooling':
            #set these params for the w token aligned case first
            max_seq_len_to_use = max_seq_len      
            max_span_width_to_use = max_span_width
            if pooling == 'none':    #align these params with sw tokens following token_reps in the case we have no pooling
                max_seq_len_to_use = max_seq_len*3          #estimate the sw tokenizer expansion, will fix this estimate later to something measured for the whole dataset, this is for pos encoding
                max_span_width_to_use = max_span_width*3    #estimate the sw tokenizer expansion, will fix this estimate later to something measured for the whole dataset, this is for pos encoding
            self.span_rep_layer = AttentionPooling_vectorized(max_seq_len=max_seq_len_to_use, 
                                                              max_span_width=max_span_width_to_use, 
                                                              **kwargs)
        else:
            raise ValueError(f'Unknown span mode {span_mode}')


    def forward(self, token_reps, w_span_ids, span_mask, sw_span_ids=None, **kwargs):
        #apply span_mask to the possible spans tensor to set start, end to 0
        #effectively they are combining the label mask and span_ids here => span_ids = 0,0 for invalid span with no label
        span_ids = w_span_ids if self.pooling else sw_span_ids
        span_ids = span_ids * span_mask.unsqueeze(-1)
        result = self.span_rep_layer(token_reps, span_ids, self.pooling, **kwargs)

        return result
    

########################################
########################################
########################################
########################################
#TESTING

def set_all_seeds(seed=42):
    random.seed(seed)                # Python random module
    np.random.seed(seed)             # Numpy module
    torch.manual_seed(seed)          # PyTorch random number generator
    torch.cuda.manual_seed(seed)     # CUDA random number generator if using GPU
    torch.cuda.manual_seed_all(seed) # CUDA random number generator for all GPUs
    torch.backends.cudnn.deterministic = True  # Makes CUDA operations deterministic
    torch.backends.cudnn.benchmark = False     # Disables CUDA convolution benchmarking for reproducibility
    os.environ['PYTHONHASHSEED'] = str(seed)  # Sets Python hash seed
    torch.use_deterministic_algorithms(True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"



def unset_deterministic():
    # Generate random seeds
    new_seed = random.randint(0, 2**32 - 1)  # Get a random seed
    # Reset random seeds to random values
    random.seed(new_seed)
    np.random.seed(new_seed)
    torch.manual_seed(new_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(new_seed)
        torch.cuda.manual_seed_all(new_seed)
    # Reset CUDA settings to defaults
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    # Remove environment variables
    os.environ.pop('PYTHONHASHSEED', None)
    os.environ.pop('CUBLAS_WORKSPACE_CONFIG', None)
    # Disable deterministic algorithms
    torch.use_deterministic_algorithms(False)




def test_attention_pooling():
    # Create test data
    set_all_seeds(42)
    hidden=4
    max_seq_len = 10
    max_span_width = 5
    token_reps = torch.randn(2, max_seq_len, hidden)
    span_ids = torch.tensor([
        [[0, 0], [2, 4], [3, 3], [3, 8]],
        [[1, 3], [2, 2], [3, 4], [2, 6]]
    ])

    # Create models
    set_all_seeds(42)
    model1 = AttentionPooling(hidden_size=hidden,
                              max_seq_len=max_seq_len,
                              max_span_width=max_span_width)
    state_dict1 = model1.state_dict()
    span_reps1 = model1(token_reps, span_ids)
    
    # Create second model
    set_all_seeds(42)
    model2 = AttentionPooling_vectorized(hidden_size=hidden,
                                         max_seq_len=max_seq_len,
                                         max_span_width=max_span_width)
    state_dict2 = model2.state_dict()
    span_reps2 = model2(token_reps, span_ids)

    # Compare all parameters
    print("\nComparing model parameters:")
    for key in state_dict1:
        is_equal = torch.allclose(state_dict1[key], state_dict2[key])
        if not is_equal:
            print(f"\nDifference in {key}:")
            print("Model 1:", state_dict1[key])
            print("Model 2:", state_dict2[key])
    

    for i in range(len(span_reps1)):
        for j in range(len(span_reps1[i])):
            print(f'model1 batch obs {i}, span {j}: {span_reps1[i, j]}')
            print(f'model2 batch obs {i}, span {j}: {span_reps2[i, j]}')


    return span_reps1, span_reps2



def test_extract_span_reps():
    # Example usage:
    batch_size = 2
    seq_len = 33
    hidden_dim = 2
    mode = 'start_end'
    #mode = 'start_inner_maxpool_end'
    #mode = 'maxpool'
    pooled = False
    seq_len = seq_len if pooled else seq_len*4

    token_reps = torch.randn(batch_size, seq_len, hidden_dim)
    w_span_ids = torch.tensor([[[0,0], [0,1],[0,2],[0,3],[0,4],[0,5],[1,1],[1,2],[1,3],[1,4],[1,5]],
                               [[0,0], [0,1],[0,2],[0,3],[0,4],[0,5],[1,1],[1,2],[1,3],[1,4],[1,5]]])
    sw_span_ids = torch.tensor([[[0,0], [0,1],[0,2],[0,3],[0,4],[0,5],[1,6],[1,7],[1,8],[1,1],[1,2],[1,3],[1,4],[1,5],[1,6],[1,7],[1,8]],
                               [[0,0], [0,1],[0,2],[0,3],[0,4],[0,5],[1,6],[1,7],[1,8],[1,1],[1,2],[1,3],[1,4],[1,5],[1,6],[1,7],[1,8]]])
    if pooled: sw_span_ids = None


    span_reps1 = extract_span_reps(token_reps, w_span_ids, sw_span_ids, mode=mode)
    span_reps2 = extract_span_reps_vectorized(token_reps, w_span_ids, sw_span_ids, mode=mode)

    print(token_reps[:,0:10,:])
    print(span_reps1)
    print(span_reps2)
    print(span_reps1==span_reps2)
    exit()
    return span_reps



# Run test
if __name__ == "__main__":
    torch.manual_seed(42)  # for reproducibility
    torch.set_printoptions(sci_mode=False)
    span_reps, span_lengths = test_attention_pooling()
    #span_reps  = test_extract_span_reps()
    #print(span_reps)



'''
So both attention pooling variants work without error, the issue is that even when setting the seed comprehesively, I get different span_rep values for each model even with different instantiations of the same model

Also ask about init the params when creatign the model, FFN, pos encoding, MHA, etc....

'''
