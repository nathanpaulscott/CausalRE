import math, random, os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.utils.rnn import pad_sequence

from .loss_functions import cross_entropy_loss, binary_cross_entropy_loss
#temp for testing here
#from loss_functions import cross_entropy_loss, binary_cross_entropy_loss


class TokenTagger(nn.Module):
    """
    Various token tagging approaches
    """
    def __init__(self, tagging_mode, **kwargs):
        super().__init__()
        self.tagging_mode = tagging_mode

        if tagging_mode == 'BECO':
            self.tagger = BECOTagger(**kwargs)
        elif tagging_mode == 'BE':
            self.tagger = BETagger(**kwargs)
        #elif tagging_mode == 'BIO':
        #    self.tagger = BIOTagger(**kwargs)
        else:
            raise ValueError(f'Unknown tagging mode {tagging_mode}')


    def forward(self, token_reps, **kwargs):
        '''
        this needs to:
        - run the token reps through a specific head depdnent on teh tag mode => output logits (b, seq_len, 2(BE) or 3(BEO/BIO))
        - make preds => make the per token preds dependent on the tag mode (argmax or decision thd) => (b, seq_len, 1(BEO/BIO) or 2(BE))
        - calc tagging loss => takes in the span labels => converts them to the tag specific labels and runs CELoss(BEO/BIO) or BCELoss(BE)
        NOTE: do this label conversion on the fly as this is just for experimentation and results, it will be a bit slower, but not so bad
        - do pred to span id conversion dependent on the tag mode.  BEO and BE will have more candidates than BIO, but the BIO conversion is done is plain python so can be slower
        - once you have your tensor of span_ids, (dynamically padded to the max num spans in the batch), you must select out the corresponding span_labels from span_labels, span_masks from span_masks
        
        '''
                
        return self.tagger(token_reps, **kwargs)




###################################################################
###################################################################
###################################################################

class BaseTagger(nn.Module):
    """
    Base class for token taggers (BECO / BE)
    Contains common span extraction and label processing methods.
    """
    def __init__(self, input_size, num_limit, max_span_width, dropout=None, **kwargs):
        super().__init__()
        self.dropout = nn.Dropout(dropout) if dropout is not None else None
        self.max_span_width = max_span_width
        self.pos_limit = num_limit
        self.neg_limit = -num_limit

    def extract_pred_span_ids_obs(self, indices):
        raise NotImplementedError("Subclasses must implement this method")

    def extract_pred_span_ids(self, token_preds):
        raise NotImplementedError("Subclasses must implement this method")

    def make_token_labels(self, token_masks, span_ids, span_masks, span_labels):
        raise NotImplementedError("Subclasses must implement this method")
   
    def forward(self, token_reps, token_masks, span_ids, span_masks, span_labels=None, force_pos_cases=True, reduction='mean', **kwargs):
        raise NotImplementedError("Subclasses must implement this method")

    
    def get_unique_indices(self, tensor):
        '''
        Returns the indices of the first occurrence of each unique row in the given tensor.
        This isnot used right now, but keep for the time being
        '''
        tensor_unique, inverse_indices = torch.unique(tensor, dim=0, return_inverse=True)
        positions = torch.arange(tensor.shape[0])
        #Use scatter_reduce to find first occurrences
        first_occurrence_idx = torch.full((tensor_unique.shape[0],), tensor.shape[0], dtype=torch.long)
        first_occurrence_idx.scatter_reduce_(0, inverse_indices, positions, reduce='amin')
        return first_occurrence_idx


    def get_unique_ids_max_scores(self, span_ids, span_scores):
        '''
        Returns:
        - `unique_span_ids`: The unique values of `span_ids` (removes duplicates).
        - `max_span_scores`: The max score among duplicates in `span_scores` for the corresponding unique row.
        unique_span_ids and max_span_scores will be aligned
        '''
        unique_span_ids, inverse_indices = torch.unique(span_ids, dim=0, return_inverse=True)
        #Get max score per unique span
        max_span_scores = torch.full((unique_span_ids.shape[0],), self.neg_limit, dtype=span_scores.dtype, device=span_scores.device)
        max_span_scores.scatter_reduce_(0, inverse_indices, span_scores, reduce='amax')
        return unique_span_ids, max_span_scores


    def extract_span_ids(self, token_preds, token_logits, span_ids, span_masks, span_labels=None, force_pos_cases=True):
        """
        Generic span extraction for both uniclass and multiclass cases.
        This has been adjusted to also output the span filter scores
        """
        batch_size = token_preds.shape[0]
        has_labels = span_labels is not None

        pred_span_ids, pred_span_scores = self.extract_pred_span_ids(token_preds, token_logits)
        out_span_ids, out_span_masks, out_span_scores, out_span_labels = [], [], [], [] if has_labels else None
        span_filter_map = []

        for i in range(batch_size):
            pred_span_ids_obs = pred_span_ids[i]
            pred_span_scores_obs = pred_span_scores[i]  # Scores for predicted spans
            all_spans_ids_obs = span_ids[i]
            all_masks_obs = span_masks[i]
            all_labels_obs = span_labels[i] if has_labels else None

            final_span_ids_obs = pred_span_ids_obs
            final_span_scores_obs = pred_span_scores_obs
            if force_pos_cases:
                pos_idx = ((all_labels_obs > 0) & all_masks_obs).nonzero(as_tuple=True)[0]
                #get the pos case span ids
                pos_span_ids_obs = all_spans_ids_obs[pos_idx] if pos_idx.numel() > 0 else torch.empty((0, 2), dtype=torch.long, device=self.device)
                #get the pos case scores, fixed at pos_limit
                pos_span_scores_obs = torch.full((pos_span_ids_obs.shape[0],), self.pos_limit, dtype=token_logits.dtype, device=self.device)
                #merge the preds with the pos cases, removing duplicates
                #Concatenate predicted and forced spans
                comb_span_ids = torch.cat((final_span_ids_obs, pos_span_ids_obs), dim=0)
                comb_span_scores = torch.cat((final_span_scores_obs, pos_span_scores_obs), dim=0)
                #Get unique span ids from comb_span_ids and the corresponding max score of the duplicates from the comb_span_scores
                final_span_ids_obs, final_span_scores_obs = self.get_unique_ids_max_scores(comb_span_ids, comb_span_scores)

            num_spans_obs = final_span_ids_obs.shape[0]
            final_span_masks_obs = torch.zeros((num_spans_obs,), dtype=torch.bool, device=self.device)
            final_span_labels_obs = torch.zeros((num_spans_obs,), dtype=torch.long, device=self.device) if has_labels else None
            
            #get dim 1 indices in all_span_ids that are in final_span_ids, put these in the span_filter_map and update the masks and labels
            span_filter_map_obs = torch.full((all_spans_ids_obs.shape[0],), -1, dtype=torch.long, device=self.device)
            if num_spans_obs > 0:
                match_matrix = (final_span_ids_obs.unsqueeze(1) == all_spans_ids_obs.unsqueeze(0)).all(dim=-1)
                final_spans_idx = match_matrix.any(dim=1).nonzero(as_tuple=True)[0]
                all_spans_idx = match_matrix[final_spans_idx].int().argmax(dim=1)
                #Update the tensor-based span_id_map
                span_filter_map_obs[all_spans_idx] = final_spans_idx
                #update the masks and labels
                final_span_masks_obs[final_spans_idx] = all_masks_obs[all_spans_idx]
                if has_labels:
                    final_span_labels_obs[final_spans_idx] = all_labels_obs[all_spans_idx]

            #append the observation tensors
            span_filter_map.append(span_filter_map_obs)
            out_span_ids.append(final_span_ids_obs)
            out_span_masks.append(final_span_masks_obs)
            out_span_scores.append(final_span_scores_obs)
            if has_labels:
                out_span_labels.append(final_span_labels_obs)
        
        #padd the list of tensors to tensors
        out_span_ids = pad_sequence(out_span_ids, batch_first=True, padding_value=0)
        out_span_masks = pad_sequence(out_span_masks, batch_first=True, padding_value=False)
        out_span_scores = pad_sequence(out_span_scores, batch_first=True, padding_value=self.neg_limit)
        out_span_labels = pad_sequence(out_span_labels, batch_first=True, padding_value=0) if has_labels else None
        span_filter_map = pad_sequence(span_filter_map, batch_first=True, padding_value=-1)

        return dict(out_span_ids    = out_span_ids,       #(batch, max_batch_filtered_spans, 2) int
                    out_span_masks  = out_span_masks,     #(batch, max_batch_filtered_spans) bool
                    out_span_scores = out_span_scores,    #(batch, max_batch_filtered_spans) float
                    out_span_labels = out_span_labels,    #(batch, max_batch_filtered_spans) int
                    span_filter_map = span_filter_map)    #(batch, all_possible_spans) int






class BECOTagger(BaseTagger):
    """
    BECO tagger => uniclass token tagger and span generator
    B = begin 0
    E = end 1
    C = B/E 2
    O = Don't care 3
    """
    def __init__(self, input_size, num_limit, max_span_width, dropout=None, **kwargs):
        super().__init__(input_size, num_limit, max_span_width, dropout, **kwargs)
        self.token_tag_head = nn.Linear(input_size, 4)
        self.B_class, self.E_class, self.C_class, self.O_class = 0, 1, 2, 3
        self.init_weights()


    def init_weights(self):
        init.xavier_uniform_(self.token_tag_head.weight)
        if self.token_tag_head.bias is not None:
            init.constant_(self.token_tag_head.bias, 0)


    def extract_pred_span_ids_obs(self, indices, logits):
        """
        Vectorized extraction of valid spans (multi-token and single-token).
        operates on a batch obs at a time
        NOTE: B/E/C_indices are actual, we only convert the ends to +1 in the final span_ids tensor
        """
        B_indices = indices['B_indices']
        E_indices = indices['E_indices']
        C_indices = indices['C_indices']
        #Merge and sort B and C (start positions), and E and C (end positions)
        BC_indices = torch.cat([B_indices, C_indices], dim=0).unique() if B_indices.numel() or C_indices.numel() else None
        EC_indices = torch.cat([E_indices, C_indices], dim=0).unique() if E_indices.numel() or C_indices.numel() else None
        #If no valid start positions, return empty tensor
        if BC_indices is None:
            return torch.empty((0, 2), dtype=torch.long, device=self.device), torch.empty((0,), dtype=logits.dtype, device=self.device)
        
        #Single-token spans: (C, C+1)
        #use C as this is specifically for tokens that are a start and end, although they may be other spans ends and begins, so this will include some single token spans that are not real, but it is ok
        #do not use B as these are for tokens that are begin only with no end, so the assicated end after it
        single_token_spans = torch.stack([C_indices, C_indices + 1], dim=1) if C_indices.numel() else torch.empty((0, 2), dtype=torch.long, device=self.device)
        single_token_scores = logits[C_indices, self.C_class] if C_indices.numel() else torch.empty((0,), dtype=logits.dtype, device=self.device)  # C logit
        #If no valid end positions, return single-token spans
        if EC_indices is None or EC_indices.numel() == 0:
            return single_token_spans, single_token_scores

        #Multi-token spans: Expand BC & EC for valid pairs
        B_exp, E_exp = BC_indices.unsqueeze(1), EC_indices.unsqueeze(0)
        #NOTE E_exp and B_exp are actual, so these comparators are good
        #if E_exp_py = E_exp + 1, i.e. actual+1, then the comparators would be E_exp_py > B_exp+1 and E_exp_py - B_exp <= max_span_width
        valid_mask = (E_exp > B_exp) & ((E_exp - B_exp) < self.max_span_width)
        # Extract indices where mask is True
        valid_B_idx, valid_E_idx = valid_mask.nonzero(as_tuple=True)
        #Generate valid multi-token spans
        #NOTE: we are converting to actual + 1 ends here
        valid_B, valid_E = BC_indices[valid_B_idx], EC_indices[valid_E_idx] + 1  # Convert E to actual+1
        multi_token_spans = torch.stack([valid_B, valid_E], dim=1) if valid_B.numel() else torch.empty((0, 2), dtype=torch.long, device=self.device)
        multi_token_scores = torch.empty((0,), dtype=logits.dtype, device=self.device)
        if multi_token_spans.numel():
            #Compute B/E scores for multi-token spans
            B_scores = logits[valid_B, self.B_class] # Get B logits at valid_B
            E_scores = logits[valid_E - 1, self.E_class] # Get E logits at valid_E-1 (actual end token)
            multi_token_scores = (B_scores + E_scores) / 2  # Average of B and E logits

        #Merge single-token and multi-token spans
        comb_span_ids = torch.empty((0, 2), dtype=torch.long, device=self.device)
        comb_span_scores = torch.empty((0,), dtype=logits.dtype, device=self.device)
        if multi_token_spans.numel() or single_token_spans.numel():
            comb_span_ids = torch.cat([multi_token_spans, single_token_spans], dim=0)
            comb_span_scores = torch.cat([multi_token_scores, single_token_scores], dim=0)
            #Get unique span ids from comb_span_ids and the corresponding max score of the duplicates from the comb_span_scores
            comb_span_ids, comb_span_scores = self.get_unique_ids_max_scores(comb_span_ids, comb_span_scores)

        return comb_span_ids, comb_span_scores


    def extract_pred_span_ids(self, token_preds, token_logits):
        """
        Extract spans from BEO predictions using efficient tensor operations.
        """
        batch_size, num_tokens = token_preds.shape

        pred_span_ids, pred_span_scores = [], []
        for i in range(batch_size):
            preds = token_preds[i]  
            logits = token_logits[i]  
            indices = dict(
                B_indices = (preds == self.B_class).nonzero(as_tuple=True)[0],  # `B` token positions  (actual)
                E_indices = (preds == self.E_class).nonzero(as_tuple=True)[0],  # `E` token positions  (actual)
                C_indices = (preds == self.C_class).nonzero(as_tuple=True)[0]   # `C` token positions  (actual)
            )
            pred_span_ids_obs, pred_span_scores_obs = self.extract_pred_span_ids_obs(indices, logits)
            pred_span_ids.append(pred_span_ids_obs)
            pred_span_scores.append(pred_span_scores_obs)
        return pred_span_ids, pred_span_scores  #ragged list of span_id tensors for the batch



    def make_token_labels(self, token_masks, span_ids, span_masks, span_labels):
        """
        Convert positive span labels into BECO token labels.
        
        Args:
            token_masks  (torch.Tensor): Shape (batch_size, num_tokens)
            span_ids (torch.Tensor): Shape (batch_size, num_spans, 2), span start-end indices.
            span_masks (torch.Tensor): Shape (batch_size, num_spans), True for valid spans.
            span_labels (torch.Tensor): Shape (batch_size, num_spans), labels (pos cases > 0).
        
        Returns:
            beo_labels (torch.Tensor): Shape (batch_size, num_tokens), BEO labels {0 (B), 1 (E), 2 (C = B/E), 3 (O)}
        """
        batch_size, num_tokens = token_masks.shape

        #Initialize beo_labels to 'O' (3) for all tokens
        token_labels = torch.full((batch_size, num_tokens), self.O_class, dtype=torch.long, device=span_ids.device)
        
        for i in range(batch_size):
            #Get valid positive spans
            pos_idx = ((span_labels[i] > 0) & span_masks[i]).nonzero(as_tuple=True)[0]  # Extract row indices
            pos_spans = span_ids[i][pos_idx]  # Shape (num_pos_cases, 2)
            if pos_spans.numel() == 0:
                continue  # Skip if no positive spans

            #Get unique B (start) and E (end) positions
            B_indices = pos_spans[:, 0].unique()  # Unique start positions
            #NOTE: E_indices are converted to actual
            E_indices = (pos_spans[:, 1] - 1).unique()  # Unique end positions (actual)
            #Find overlapping positions where B and E coincide
            C_indices = B_indices[torch.isin(B_indices, E_indices)]  # Find overlapping positions
            # Assign labels
            token_labels[i][B_indices] = self.B_class
            token_labels[i][E_indices] = self.E_class
            token_labels[i][C_indices] = self.C_class

        #Ensure padding tokens remain as 'O'
        token_labels = torch.where(token_masks, token_labels, torch.tensor(self.O_class, dtype=torch.long, device=self.device))
        
        return token_labels  # Shape (batch_size, num_tokens)


    def forward(self, token_reps, token_masks, span_ids, span_masks, span_labels=None, force_pos_cases=True, reduction='mean', **kwargs):
        self.device = token_reps.device
        tagger_loss = 0

        #Get the BECO logits for each token
        if self.dropout is not None:
            token_reps = self.dropout(token_reps)
        token_logits = self.token_tag_head(token_reps)    # Shape: (batch, num_tokens, 4)

        #Get the token preds
        token_preds = torch.argmax(token_logits, dim=-1)  # Get raw predictions
        #apply the token mask to set masked out tokens to -1, DO NOT just mult the mask wih the preds, it would not work in this case
        token_preds = torch.where(token_masks, token_preds, torch.tensor(-1, device=self.device))
        #get the filtered span tensors from the preds and labels
        #results will contain: out_span_ids, out_span_masks, out_span_labels, out_span_scores
        result = self.extract_span_ids(token_preds, token_logits, span_ids, span_masks, span_labels, force_pos_cases)

        # Compute loss if labels are provided
        if span_labels is not None:
            token_labels = self.make_token_labels(token_masks, span_ids, span_masks, span_labels)
            tagger_loss = cross_entropy_loss(token_logits, token_labels, token_masks, reduction=reduction)

        #add the tagger_loss
        result['tagger_loss'] = tagger_loss
        return result
    

#################################################################################################################
#################################################################################################################
#################################################################################################################


class BETagger(BaseTagger):
    """
    BE tagger => multiclass token tagger and span generator
    B = begin 0
    E = end 1
    """
    def __init__(self, input_size, num_limit, max_span_width, dropout=None, predict_thd=0.3, **kwargs):
        super().__init__(input_size, num_limit, max_span_width, dropout, **kwargs)
        self.token_tag_head = nn.Linear(input_size, 2)
        self.predict_thd = predict_thd
        self.B_class = 0
        self.E_class = 1
        self.init_weights()


    def init_weights(self):
        init.xavier_uniform_(self.token_tag_head.weight)
        if self.token_tag_head.bias is not None:
            init.constant_(self.token_tag_head.bias, 0)


    def extract_pred_span_ids_obs(self, indices, logits):
        """
        Vectorized extraction of valid spans.
        operates on a batch obs at a time
        NOTE: B/E are actual, we only convert the ends to +1 in the final span_ids tensor
        NOTE: B/E are derived from multiclass preds, so single token spans are only when we get a B and and E on the same token
        Thus one of our conditionals is different:
        - E_exp >= B_exp, i.e it can fall on the same token for single token spans in this case
        """
        B_indices = indices['B_indices']
        E_indices = indices['E_indices']
        if B_indices.numel() == 0 or E_indices.numel() == 0:
            return torch.empty((0, 2), dtype=torch.long, device=self.device), torch.empty((0,), dtype=logits.dtype, device=self.device)
        B_exp, E_exp = B_indices.unsqueeze(1), E_indices.unsqueeze(0)
        #the valid mask conditionals handle single token spans => (E_exp >= B_exp) instead of (E_exp > B_exp)
        #the width conditional is the same as the uniclass case
        valid_mask = (E_exp >= B_exp) & ((E_exp - B_exp) < self.max_span_width)
        valid_B_idx, valid_E_idx = valid_mask.nonzero(as_tuple=True)
        #Generate valid spans
        #NOTE: we are converting to actual + 1 ends here
        valid_B, valid_E = B_indices[valid_B_idx], E_indices[valid_E_idx] + 1
        
        pred_span_ids = torch.empty((0, 2), dtype=torch.long, device=self.device)
        pred_span_scores = torch.empty((0,), dtype=logits.dtype, device=self.device)
        if valid_B.numel():
            #make the span_ids
            pred_span_ids = torch.stack([valid_B, valid_E], dim=1)
            #Compute span scores as the mean of B and E logits
            B_scores = logits[valid_B_idx, 0]  # Get B logits at valid_B_idx
            E_scores = logits[valid_E_idx, 1]  # Get E logits at valid_E_idx
            pred_span_scores = (B_scores + E_scores) / 2  # Average of B and E logits
            #run unique on span_ids and take the max score from span_scores
            pred_span_ids, pred_span_scores = self.get_unique_ids_max_scores(pred_span_ids, pred_span_scores)

        return pred_span_ids, pred_span_scores



    def extract_pred_span_ids(self, token_preds, token_logits):
        """
        Extract spans from BE predictions
        """
        batch_size, num_tokens, _ = token_preds.shape

        pred_span_ids, pred_span_scores = [], []
        for i in range(batch_size):
            preds = token_preds[i]
            logits = token_logits[i]
            indices = dict(
                B_indices = (preds[:, 0] > self.predict_thd).nonzero(as_tuple=True)[0],  # `B` token positions (actual)
                E_indices = (preds[:, 1] > self.predict_thd).nonzero(as_tuple=True)[0]   # `E` token positions (actual)
            )
            pred_span_ids_obs, pred_span_scores_obs = self.extract_pred_span_ids_obs(indices, logits)
            pred_span_ids.append(pred_span_ids_obs)
            pred_span_scores.append(pred_span_scores_obs)
        return pred_span_ids, pred_span_scores  #ragged list of span_id tensors for the batch



    def make_token_labels(self, token_masks, span_ids, span_masks, span_labels):
        """
        Convert span labels into BE token labels.
        """
        batch_size, num_tokens = token_masks.shape
        token_labels = torch.zeros((batch_size, num_tokens, 2), dtype=torch.float, device=self.device)

        for i in range(batch_size):
            pos_idx = ((span_labels[i] > 0) & span_masks[i]).nonzero(as_tuple=True)[0]
            pos_spans = span_ids[i][pos_idx]
            if pos_spans.numel() == 0:
                continue
            #B_indices is actual
            B_indices = pos_spans[:, 0].unique()
            #E_indices is actual
            E_indices = (pos_spans[:, 1] - 1).unique()

            token_labels[i, B_indices, 0] = 1.0
            token_labels[i, E_indices, 1] = 1.0

        #apply the token_masks, can do the standard way here as a masked out label should be 0
        token_labels = token_labels * token_masks.unsqueeze(-1)
        return token_labels


    def forward(self, token_reps, token_masks, span_ids, span_masks, span_labels=None, force_pos_cases=True, reduction='mean', **kwargs):
        self.device = token_reps.device
        tagger_loss = 0

        #get BE logits for each token
        if self.dropout is not None:
            token_reps = self.dropout(token_reps)
        token_logits = self.token_tag_head(token_reps)  # Shape: (batch, num_tokens, 2)   float
        token_probs = torch.sigmoid(token_logits)  # Convert logits to probabilities    (batch, num_tokens, 2)    float
        token_preds = token_probs > self.predict_thd  # Threshold predictions    (batch, num_tokens, 2) bool
        #apply the token mask to set masked out tokens to 0,0

        token_preds = token_preds * token_masks.unsqueeze(-1)

        #results will contain: out_span_ids, out_span_masks, out_span_labels, out_span_scores
        result = self.extract_span_ids(token_preds, token_logits, span_ids, span_masks, span_labels, force_pos_cases)

        # Compute loss if labels are provided
        if span_labels is not None:
            token_labels = self.make_token_labels(token_masks, span_ids, span_masks, span_labels)
            tagger_loss = binary_cross_entropy_loss(token_logits, token_labels, token_masks, reduction=reduction)

        #add the tagger_loss
        result['tagger_loss'] = tagger_loss
        return result









# ========================
# TEST CODE (Paste Below)
# ========================
#not working, need to revise this shit....
#span scores change has fucked it.....
#maybe it is taking the first duplicate idx, but in teh span scores that is not necessarily the pos case one
#think about how to fix that


if __name__ == "__main__":
    import torch

    print("\n🚀 Running Full BEOTagger Test...\n")

    # Define batch size and number of tokens
    batch_size = 2
    num_tokens = 6
    num_spans = 3
    hidden = 10
    max_span_width = 4
    # Initialize the model
    torch.manual_seed(42)
    #model = BECOTagger(input_size=hidden, num_limit=1e38, max_span_width=max_span_width, dropout=0.1)
    model = BETagger(input_size=hidden, num_limit=1e38, max_span_width=max_span_width, dropout=0.1, predict_thd=0.3)
    print("✅ Model initialized successfully!")

    #Random token embeddings
    torch.manual_seed(42)
    token_reps = torch.randn(batch_size, num_tokens, hidden)
    
    # Token masks (True = valid, False = padding)
    token_masks = torch.tensor([[True, True, True, True, True, False], 
                                [True, True, True, True, True, False]])
    # Random span IDs (start, end) within valid token range
    span_ids = torch.tensor([
        [[0, 3], [3, 5], [2, 3]],
        [[0, 1], [2, 3], [3, 5]]
    ])
    # Span masks (True = valid, False = ignored)
    span_masks = torch.tensor([
        [True, True, True],
        [True, False, True]
    ])
    # Random span labels (0 = no span, >0 valid span)
    span_labels = torch.tensor([
        [3, 0, 8],
        [2, 3, 4]
    ])

    # Run full model forward pass
    result = model(token_reps, token_masks, span_ids, span_masks, span_labels, force_pos_cases=False)

    # Assertions to check all expected outputs exist
    assert "out_span_ids" in result, "❌ Missing out_span_ids!"
    assert "out_span_masks" in result, "❌ Missing out_span_masks!"
    assert "out_span_labels" in result, "❌ Missing out_span_labels!"
    assert "tagger_loss" in result, "❌ Missing tagger_loss!"

    # Ensure output shapes match batch size
    assert result["out_span_ids"].shape[0] == batch_size, "❌ out_span_ids shape mismatch!"
    assert result["out_span_masks"].shape[0] == batch_size, "❌ out_span_masks shape mismatch!"
    assert result["out_span_labels"].shape[0] == batch_size, "❌ out_span_labels shape mismatch!"

    #Ensure padding tokens remain as 'O' (3) => only for the BECO tagger
    #token_labels = model.make_token_labels(token_masks, span_ids, span_masks, span_labels)
    #assert token_labels.masked_select(~token_masks).eq(3).all(), "❌ Padding tokens are not correctly marked as 'O' (2)"

    print("\n🎉 ALL TESTS PASSED SUCCESSFULLY!\n")


    print(f"out_span_ids: {result['out_span_ids']}")
    print(f"out_span_masks: {result['out_span_masks']}")
    print(f"out_span_labels: {result['out_span_labels']}")
    print(f"out_span_scores: {result['out_span_scores']}")
    print(f"tagger_loss: {result['tagger_loss']}")