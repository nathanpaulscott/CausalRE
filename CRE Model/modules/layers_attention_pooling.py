import torch
import torch.nn.functional as F
from torch import nn
import torch.nn.init as init
import numpy as np
import math, random, os


from .layers_other import PositionalEncoding, FFNProjectionLayer


'''
split this code out from the span_reps code, have still not settled on the way to do it
ultimately it could be used for anything, so take out the span specific nature of it

To do....
'''




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
        
        #init weights for the dummy query only
        self.init_weights()


    def init_weights(self):
        # Init dummy query
        init.xavier_normal_(self.dummy_query)


    def forward(self, token_reps, span_ids, span_masks, pooling, **kwargs):
        '''
        Args:
            token_reps: (batch, max_batch_seq_len, hidden)    #sw or word token aligned
            span_ids is of shape    (batch, num_spans, 2)       where num_spans = w_seq_len*max_span_width, values in last dim are w/sw aligned based on pooling
            span_masks is of shape   (batch, num_spans)          where num_spans = w_seq_len*max_span_width
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
        for obs_id in range(batch):
            span_reps = []
            for span_id in range(num_spans):
                start, end = span_ids[obs_id, span_id]
                # Handle invalid spans
                if span_masks[obs_id, span_id] == 0:
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
    



class AttentionPooling_vectorized_old(nn.Module):
    '''
    KEEP THIS VERSION FOR NOW, NOT SURE IF IT IS BETTER OR NOT, DECIDE LATER
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
        
        #init weights for the dummy query only
        self.init_weights()

    def init_weights(self):
        # Init dummy query
        init.xavier_normal_(self.dummy_query)

    def forward(self, token_reps, span_ids, span_masks, pooling, **kwargs):
        '''
        Args:
            token_reps: (batch, max_batch_seq_len, hidden)    #sw or word token aligned
            span_ids is of shape    (batch, num_spans, 2)       where num_spans = w_seq_len*max_span_width, values in last dim are w/sw aligned based on pooling
            span_masks is of shape   (batch, num_spans)          where num_spans = w_seq_len*max_span_width
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

        # Get maximum span length and apply the span_masks
        span_lengths = (span_ids[:, :, 1] - span_ids[:, :, 0]) * span_masks # (batch, num_spans)
        max_span_len = max(span_lengths.max().item(), 1)
        # Initialize tensors for batched processing
        span_token_reps = torch.zeros((batch_size, num_spans, max_span_len, hidden), device=token_reps.device)
        key_padding_mask = torch.ones((batch_size, num_spans, max_span_len), device=token_reps.device, dtype=torch.bool)  # True means ignore position
        # Extract span tokens and create attention masks
        for b in range(batch_size):
            for s in range(num_spans):
                start, end = span_ids[b, s]
                span_len = span_lengths[b, s]
                if span_len > 0:
                    # Copy span tokens
                    span_token_reps[b, s, :span_len, :] = token_reps[b, start:end, :]
                    # Update attention mask for valid tokens
                    key_padding_mask[b, s, :span_len] = False

        # Add span positional encodings if enabled
        if self.use_span_pos_encoding:
            span_token_reps = self.span_pos_encoder(span_token_reps)
        
        # Reshape tensors for batched attention - keeping each span separate
        # Each dummy-span pair becomes a separate "batch" item
        dummy_query = self.dummy_query.expand(batch_size * num_spans, 1, hidden)
        span_token_reps = span_token_reps.view(batch_size * num_spans, max_span_len, hidden)
        key_padding_mask = key_padding_mask.view(batch_size * num_spans, max_span_len)

        # Apply attention with masking
        span_reps, _ = self.attn(
            dummy_query,
            span_token_reps,
            span_token_reps,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        # Apply layer norm and FFN, can do it here or after reshaping, doesn't matter
        span_reps = self.norm(span_reps + dummy_query)
        span_reps = self.norm(self.ffn(span_reps) + span_reps)

        # Reshape
        span_reps = span_reps.view(batch_size, num_spans, hidden)  # (batch, num_spans, hidden)
        #Zero out masked spans
        span_reps[~span_masks] = 0
        
        return span_reps  # (batch, num_spans, hidden)



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
        
        #init weights for the dummy query only
        self.init_weights()


    def init_weights(self):
        # Init dummy query
        init.xavier_normal_(self.dummy_query)


    def forward(self, token_reps, span_ids, span_masks, pooling, **kwargs):
        '''
        Vectorized attention pooling for span representations.
        
        Args:
            token_reps: (batch, max_batch_seq_len, hidden)
            span_ids: (batch, num_spans, 2) - Start and end+1 indices for spans
            span_masks: (batch, num_spans) - Boolean mask indicating valid spans
            pooling: True for word token aligned reps

        Returns:
            span_reps: (batch, num_spans, hidden)
        '''
        batch_size, seq_len, hidden = token_reps.shape
        batch_size, num_spans, _ = span_ids.shape
        # Add positional encodings to the full sequence
        token_reps = self.seq_pos_encoder(token_reps)
        #Compute span lengths, use the span_masks to zero out masked out spans, will be used later to make the key_padding_mask
        span_lengths = (span_ids[:, :, 1] - span_ids[:, :, 0]) * span_masks      #shape (batch, num_spans)
        max_span_len = max(span_lengths.max().item(), 1)                        #scalar
        # Create key_padding_mask (== 1 for masked out spans)
        range_tensor = torch.arange(max_span_len, device=token_reps.device)     #shape (seq_len)
        expanded_lengths = span_lengths.unsqueeze(-1)                           #(batch, num_spans, 1)
        key_padding_mask = range_tensor >= expanded_lengths                     #True means masked out (batch, num_spans, seq_len)
        # Vectorized token gathering
        span_starts = span_ids[..., 0].unsqueeze(-1)                            #(batch, num_spans, 1)
        indices = span_starts + range_tensor                                    #NOt sure (batch, num_spans, max_span_len)
        gather_indices = indices.unsqueeze(-1).expand(-1, -1, -1, hidden)
        span_token_reps = torch.gather(                                         #(batch, num_spans, seq_len, hidden)
            token_reps.unsqueeze(1).expand(-1, num_spans, -1, -1),
            dim=2,
            index=gather_indices
        )
        # Mask out invalid positions
        span_token_reps = span_token_reps.masked_fill(key_padding_mask.unsqueeze(-1), 0)

        # Add span positional encodings if enabled
        if self.use_span_pos_encoding:
            span_token_reps = self.span_pos_encoder(span_token_reps)

        # Reshape for batched attention
        dummy_query = self.dummy_query.expand(batch_size * num_spans, 1, hidden)
        span_token_reps = span_token_reps.view(batch_size * num_spans, max_span_len, hidden)
        key_padding_mask = key_padding_mask.view(batch_size * num_spans, max_span_len)

        # Apply multi-head attention
        span_reps, _ = self.attn(
            dummy_query,
            span_token_reps,
            span_token_reps,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        span_reps = self.norm(span_reps + dummy_query)
        span_reps = self.norm(self.ffn(span_reps) + span_reps)

        #Reshape
        span_reps = span_reps.view(batch_size, num_spans, hidden)
        #apply the span_masks to set unwanted span_reps to all 0
        span_reps[~span_masks] = 0

        return span_reps


