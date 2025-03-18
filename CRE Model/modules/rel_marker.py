import torch
from torch import nn






class RelMarker(nn.Module):
    '''
    This is code to mark rels in the original token sequence and re-run through the bert encoder to get enriched representations
    the enriched rep for each rel is the cls token
    '''
    def __init__(self, transformer_encoder, config):
        super().__init__()

        self.config = config
        self.transformer_encoder = transformer_encoder

        #Projection layer (concat start/end embeddings back to hidden size)
        if self.config.rel_marker_run_option == 'marker':
            self.projection = nn.Linear(self.config.hidden_size * 4, self.config.hidden_size)
        elif self.config.rel_marker_run_option == 'all':
            self.projection = nn.Linear(self.config.hidden_size * 5, self.config.hidden_size)



    def mark_rels_batch(self, tokens, rel_ids, span_ids, rel_masks):
        """
        Marks each relation with special tokens <head/tail_start> and <head/tail_end>.
        
        Args:
            tokens: List[List[str]] (batch of unpadded word-tokenized sequences)
            rel_ids: Tensor (b, top_k_rels, 2), containing head and tail span indices
            span_ids: Tensor (b, top_k_spans, 2), containing span boundaries (pythonic end)
            rel_masks: Tensor (b, top_k_rels), indicating valid relations
            
        Returns:
            marked_sequences: List[List[str]] (num_valid_rels_in_batch)
        """
        # Read the special marker tokens from the transformer encoder
        head_start_token = self.transformer_encoder.head_start_token
        head_end_token = self.transformer_encoder.head_end_token
        tail_start_token = self.transformer_encoder.tail_start_token
        tail_end_token = self.transformer_encoder.tail_end_token

        rel_ids_list = rel_ids.tolist()
        span_ids_list = span_ids.tolist()
        rel_masks_list = rel_masks.tolist()

        marked_sequences = []
        for tokens_obs, rel_id_obs, rel_mask_obs, span_id_obs in zip(tokens, rel_ids_list, rel_masks_list, span_ids_list):    
            for (head_idx, tail_idx), is_valid in zip(rel_id_obs, rel_mask_obs):    
                if is_valid:
                    # Get the start/end positions
                    head_start, head_end = span_id_obs[head_idx]
                    tail_start, tail_end = span_id_obs[tail_idx]

                    # Check for validity
                    if (head_start >= head_end or head_end > len(tokens_obs) or tail_start >= tail_end or tail_end > len(tokens_obs)):
                        raise Exception(f'Invalid unmasked relation detected: '
                                        f'head_start={head_start}, head_end={head_end}, '
                                        f'tail_start={tail_start}, tail_end={tail_end}, '
                                        f'len(tokens)={len(tokens_obs)}')

                    # Collect all positions and markers
                    markers = [
                        (head_start, head_start_token, 'start'),
                        (head_end, head_end_token, 'end'),
                        (tail_start, tail_start_token, 'start'),
                        (tail_end, tail_end_token, 'end')
                    ]

                    # Sort markers, considering 'start' tokens should come after 'end' if at the same position
                    markers.sort(key=lambda x: (x[0], x[2] == 'end'))

                    # Insert markers into the sequence
                    marked_sequence = []
                    prev_pos = 0
                    for pos, marker, type in markers:
                            marked_sequence += tokens_obs[prev_pos:pos] + [marker]
                            prev_pos = pos
                    marked_sequence += tokens_obs[prev_pos:]

                    #append to the marked_sequences list
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

        if self.config.rel_marker_run_option == 'cls':
            return cls_emb
        
        # Read special token IDs from transformer_encoder
        head_start_id = self.transformer_encoder.head_start_id
        head_end_id = self.transformer_encoder.head_end_id
        tail_start_id = self.transformer_encoder.tail_start_id
        tail_end_id = self.transformer_encoder.tail_end_id
        # Find positions of the special tokens
        head_start_pos = (input_ids == head_start_id).nonzero(as_tuple=False)
        head_end_pos = (input_ids == head_end_id).nonzero(as_tuple=False)
        tail_start_pos = (input_ids == tail_start_id).nonzero(as_tuple=False)
        tail_end_pos = (input_ids == tail_end_id).nonzero(as_tuple=False)
        # Extract embeddings for <span_start> and <span_end>
        head_start_emb = embeddings[head_start_pos[:, 0], head_start_pos[:, 1]]
        head_end_emb = embeddings[head_end_pos[:, 0], head_end_pos[:, 1]]
        tail_start_emb = embeddings[tail_start_pos[:, 0], tail_start_pos[:, 1]]
        tail_end_emb = embeddings[tail_end_pos[:, 0], tail_end_pos[:, 1]]
        #concat
        rel_emb = torch.cat([head_start_emb, head_end_emb, tail_start_emb, tail_end_emb], dim=-1)
        if self.config.rel_marker_run_option == 'all':
            # Concatenate <span_start>, <span_end>, and CLS embeddings
            rel_emb = torch.cat([rel_emb, cls_emb], dim=-1)

        return rel_emb  # Shape: (batch*k, hidden*x)   x is 5 for all, 4 for marker and 1 for cls



    def forward(self, tokens, rel_ids, span_ids, rel_masks):
        """
        tokens_batch: List[List[str]], len=batch_size, unpadded word-tokenized sequences
        span_ids: Tensor of shape (batch_size, top_k_spans, 2), pythonic indices
        span_masks: Tensor of shape (batch_size, top_k_spans), bool masks for valid spans
        
        Returns:
            span_reps: Tensor (batch_size, top_k_spans, hidden_size)
        """
        batch, top_k_rels, _ = rel_ids.shape

        #Mark tokens with span markers; returns flattened list of marked sequences
        marked_sequences = self.mark_rels_batch(tokens, rel_ids, span_ids, rel_masks)

        #Run marked sequences through transformer encoder
        result = self.transformer_encoder.transformer_encoder_basic(marked_sequences, type='rel', window=self.config.rel_marker_window)
        embeddings = result['embeddings']  # (batch_size*valid_k, marked_seq_len, hidden)
        input_ids = result['input_ids']    # (batch_size*valid_k, marked_seq_len)

        #Extract embeddings from special tokens
        rel_reps = self.extract_special_embeddings(embeddings, input_ids)  # (batch_size*valid_k, hidden*x)
        #Project embeddings back to hidden dimension
        if self.config.rel_marker_run_option in ['marker', 'all']:
            rel_reps = self.projection(rel_reps)  # (batch_size*valid_k, hidden)

        #Reshape to (batch_size, top_k_spans, hidden)
        #Reconstruct the original shape with padding for invalid spans
        output_rel_reps = torch.zeros(batch, top_k_rels, rel_reps.size(-1), device=self.config.device)
        valid_positions = rel_masks.view(-1).nonzero(as_tuple=False).squeeze(1)
        output_rel_reps.view(-1, rel_reps.size(-1))[valid_positions] = rel_reps

        return output_rel_reps  # (batch, top_k_rels, hidden)
