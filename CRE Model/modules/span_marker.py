import torch
from torch import nn




'''
ok, so think about invalid spans, either invalid (0,0) or masked.  It seems to be handling it in a fucked up way

'''




class SpanMarker(nn.Module):
    '''
    This is code to mark spans in the original token sequence and re-run through the bert encoder to get enriched representations
    the enriched rep for each span is the reporojected concat of the span_start and span_end marker tokens
    '''
    def __init__(self, transformer_encoder, config):
        super().__init__()

        self.config = config
        self.transformer_encoder = transformer_encoder

        #Projection layer (concat start/end embeddings back to hidden size)
        if self.config.span_marker_run_option == 'marker':
            self.projection = nn.Linear(self.config.hidden_size * 2, self.config.hidden_size)
        elif self.config.span_marker_run_option == 'all':
            self.projection = nn.Linear(self.config.hidden_size * 3, self.config.hidden_size)



    def mark_spans_batch(self, tokens, span_ids, span_masks):
        """
        Marks each span with special tokens <span_start> and <span_end>.
        
        Args:
            tokens: List[List[str]] (batch of unpadded word-tokenized sequences)
            span_ids: Tensor (b, top_k_spans, 2) with pythonic end indices
            span_masks: Tensor (b, top_k_spans), indicating valid spans
            
        Returns:
            marked_sequences: List[List[str]] of length (num_valid_spans_in_batch)
        """
        #read the special marker tokens from the transformer encoder
        start_token = self.transformer_encoder.span_start_token
        end_token = self.transformer_encoder.span_end_token

        marked_sequences = []
        span_ids_list = span_ids.tolist()
        span_masks_list = span_masks.tolist()
        for tokens_obs, spans_obs, masks_obs in zip(tokens, span_ids_list, span_masks_list):    #iterate through each obs in the batch
            for (start, end), is_valid in zip(spans_obs, masks_obs):    #iterate through each span in the obs
                if is_valid:   #if the span is not masked
                    #do quick validity checks and throw if fails, this shoudl never fail , but leave in for now
                    if start >= end or end > len(tokens_obs):
                        raise Exception(f'Invalid unmasked span detected: start={start}, end={end}, len(tokens)={len(tokens_obs)}')
                    #mark the tokens
                    marked_sequence = tokens_obs[:start] + [start_token] + tokens_obs[start:end] + [end_token] + tokens_obs[end:]
                    marked_sequences.append(marked_sequence)

        return marked_sequences



    def extract_special_embeddings(self, embeddings, input_ids):
        """
        Extracts embeddings for <span_start>, <span_end>, and [CLS] tokens.
        
        Args:
            embeddings: Tensor of shape (batch*k, seq_len, hidden)
            input_ids: Tensor of shape (batch*k, seq_len)
        
        Returns:
            span_emb: Tensor of shape (batch*k, hidden*3)
        """
        # Extract CLS embeddings (always at position 0)
        cls_emb = embeddings[:, 0, :]  # CLS token embedding

        if self.config.span_marker_run_option == 'cls':
            return cls_emb
        
        # Read special token IDs from transformer_encoder
        start_id = self.transformer_encoder.span_start_id
        end_id = self.transformer_encoder.span_end_id
        # Find positions of the special tokens
        span_start_pos = (input_ids == start_id).nonzero(as_tuple=False)
        span_end_pos = (input_ids == end_id).nonzero(as_tuple=False)
        # Extract embeddings for <span_start> and <span_end>
        span_start_emb = embeddings[span_start_pos[:, 0], span_start_pos[:, 1]]
        span_end_emb = embeddings[span_end_pos[:, 0], span_end_pos[:, 1]]
        #concat
        span_emb = torch.cat([span_start_emb, span_end_emb], dim=-1)
        if self.config.span_marker_run_option == 'all':
            # Concatenate <span_start>, <span_end>, and CLS embeddings
            span_emb = torch.cat([span_emb, cls_emb], dim=-1)

        return span_emb  # Shape: (batch*k, hidden*x)     x is 3 for all, 2 for marker and 1 for cls
  


    def forward(self, tokens, span_ids, span_masks):
        """
        tokens_batch: List[List[str]], len=batch_size, unpadded word-tokenized sequences
        span_ids: Tensor of shape (batch_size, top_k_spans, 2), pythonic indices
        span_masks: Tensor of shape (batch_size, top_k_spans), bool masks for valid spans
        
        Returns:
            span_reps: Tensor (batch_size, top_k_spans, hidden_size)


        IDEA
        It looks like the cls token is the best rep to extract and classify as opposed to the markers
        a potential adjustment for long seq is to modify the attention mask to a window around the marked span
        so the cls doesn't get diluted by text far away from the span
        can try that later..... 
        """
        batch_size, top_k_spans, _ = span_ids.shape

        #Mark tokens with span markers; returns flattened list of marked sequences
        marked_sequences = self.mark_spans_batch(tokens, span_ids, span_masks)

        #Run marked sequences through transformer encoder
        result = self.transformer_encoder.transformer_encoder_basic(marked_sequences, type='span', window=self.config.span_marker_window)
        embeddings = result['embeddings']  # (batch_size*valid_k, marked_seq_len, hidden)
        input_ids = result['input_ids']    # (batch_size*valid_k, marked_seq_len)

        #Extract embeddings from special tokens
        span_reps = self.extract_special_embeddings(embeddings, input_ids)  # (batch_size*valid_k, hidden*x)
        #Project embeddings back to hidden dimension
        if self.config.span_marker_run_option in ['marker', 'all']:
            span_reps = self.projection(span_reps)  # (batch_size*valid_k, hidden)

        #Reshape to (batch_size, top_k_spans, hidden)
        #Reconstruct the original shape with padding for invalid spans
        output_span_reps = torch.zeros(batch_size, top_k_spans, span_reps.size(-1), device=self.config.device)
        valid_positions = span_masks.view(-1).nonzero(as_tuple=False).squeeze(1)
        output_span_reps.view(-1, span_reps.size(-1))[valid_positions] = span_reps

        return output_span_reps  # (batch_size, top_k_spans, hidden)
