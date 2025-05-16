from collections import defaultdict
from typing import Tuple, List, Dict, Union

import torch
from torch.utils.data import DataLoader
import random



class RelationProcessor():
    '''
    We run this from the model forward pass as we need the filtered spans as inputs.  
    We can't do it on all possible spans as its a quadratic expansion of the number of candidate spans
    It works with the rel annotation data from `x['relations']` to form rel labels, masks and ids tensors based on candidate span IDs.
    '''
    def __init__(self, config):
        '''
        This just accepts the config namespace object
        '''
        self.config = config
        self.has_labels = self.config.run_type == 'train'



    def init_rel_labels(self, batch, num_spans, device):
        '''
        makes the blank rel_labels tensors
        '''
        if self.config.rel_labels == 'unilabel':
            return torch.zeros((batch, num_spans, num_spans), dtype=torch.long, device=device)
        elif self.config.rel_labels == 'multilabel':
            return torch.zeros((batch, num_spans, num_spans, self.config.num_rel_types), dtype=torch.bool, device=device)


    def update_rel_labels(self, rel_labels, batch_idx, head_cand_idx, tail_cand_idx, rel_type):
        '''
        updates the rel_labels tensor with a label id for one position in the tensor
        '''
        r_label_id = self.config.r_to_id[rel_type]
        if self.config.rel_labels == 'unilabel':
            rel_labels[batch_idx, head_cand_idx, tail_cand_idx] = r_label_id
        elif self.config.rel_labels == 'multilabel':
            #convert rel label id to a position in the multilabel binary label vector
            #NOTE: the r_label_id for the multilabel case will be the position in the vector as idx 0 is the first pos class
            rel_labels[batch_idx, head_cand_idx, tail_cand_idx, r_label_id] = True


    def flatten_rel_labels(self, rel_labels):
        '''
        converts the rel_labels from:
        unilabel => (batch, num_spans, num_spans) => (batch, num_spans**2)
        multilabel => (batch, num_spans, num_spans, num rel types) => (batch, num_spans**2, num rel types)
        '''
        batch = rel_labels.shape[0]
        if self.config.rel_labels == 'unilabel':
            return rel_labels.view(batch, -1)
        elif self.config.rel_labels == 'multilabel':
            return rel_labels.view(batch, -1, self.config.num_rel_types)



    def update_lost_rel_data(self, all_span_ids, orig_map, token_logits, filter_score_span, head_idx_raw, tail_idx_raw, head_cand_idx, tail_cand_idx, lost_rel_penalty, lost_rel_counts, b):
        '''
        calculates the lost rel penalty for one relation       
        '''
        eps = 1e-8
        if self.config.token_tagger and self.config.tagging_mode == 'BE':
            #get the start/end actual of the head/tail span
            head_start, head_end = all_span_ids[b, orig_map[b][head_idx_raw]]
            tail_start, tail_end = all_span_ids[b, orig_map[b][tail_idx_raw]]
            #convert ends to actual
            head_end = head_end - 1
            tail_end = tail_end - 1
            #make the scores
            B_score_h = token_logits[b, head_start, 0]
            E_score_h = token_logits[b, head_end,   1]
            h_label_score = (B_score_h + E_score_h) / 2
            B_score_t = token_logits[b, tail_start, 0]
            E_score_t = token_logits[b, tail_end,   1]
            t_label_score = (B_score_t + E_score_t) / 2
            #make the penalties
            h_penalty = -torch.log(torch.sigmoid(h_label_score) + eps)
            t_penalty = -torch.log(torch.sigmoid(t_label_score) + eps)

        elif self.config.span_filtering_type == 'bfhs':
            #just use the filter span scores for the penalty
            # use binary filter head score for BFHS
            h_score = filter_score_span[b, orig_map[b][head_idx_raw]]
            t_score = filter_score_span[b, orig_map[b][tail_idx_raw]]
            h_penalty = -torch.log(torch.sigmoid(h_score) + eps)
            t_penalty = -torch.log(torch.sigmoid(t_score) + eps)
        
        else:
            raise Exception('not implemented yet for both or tths:BECO, current tagger flag: {self.config.token_tagger}, current token tagging mode: {self.config.tagging_mode}')
        
        if head_cand_idx == -1 and tail_cand_idx != -1:
            #increment the lost rel counter for this batch
            lost_rel_counts[b] += 1
            #add the lost_rel_penalty
            #lost_rel_penalty.add(self.config.lost_rel_penalty_incr)    #this doesn't backpropagate
            lost_rel_penalty = lost_rel_penalty + h_penalty
        elif head_cand_idx != -1 and tail_cand_idx == -1:
            #increment the lost rel counter for this batch
            lost_rel_counts[b] += 1
            #add the lost_rel_penalty
            #lost_rel_penalty.add(self.config.lost_rel_penalty_incr)    #this doesn't backpropagate
            lost_rel_penalty = lost_rel_penalty + t_penalty
        else:
            #increment the lost rel counter again as it has double missing spans
            lost_rel_counts[b] += 2
            #add the lost_rel_penalty again for double missing spans
            #lost_rel_penalty.add(2 * self.config.lost_rel_penalty_incr)  #this doesn't backpropagate
            lost_rel_penalty = lost_rel_penalty + h_penalty + t_penalty

        return lost_rel_penalty, lost_rel_counts



    def get_rel_labels(self, rel_annotations, orig_map, span_filter_map, all_span_ids, span_masks, token_logits, filter_score_span):
        '''
        ok so if mode is tths or both, token_logits will be not None, if we are in bfhs then it will be None
        if it is None, we use the filter_score_spans as they will be for all possible spans
        So we have to lookup the missed span in teh all span ids list which we get to with just orig_map

        if it is not None, then we use the token_logits and have to generate the score for all the missed spans
        The calc will be depending on wheterh we are using BE or BECO for the tagger loss, we can determine this by the last dim of the token_logits BE = 2, BECO = 4
        So we need the start/end-1 from teh span annotations, I think orig_map[b][head_idx_raw], orig_map[b][tail_idx_raw]
        Note sure the input is the annoattion head and tail id, we need orig map to map this to the all span ids idx, then we read the start/end idx, then we lookup the logit from token_logits with start and end-1
        for BE:
            #dim 0 is the batch and dim 1 is the token idx and dim 2 is the B or E
            B_scores = logits[valid_B_idx, 0]  # Get B logits at valid_B_idx
            E_scores = logits[valid_E_idx, 1]  # Get E logits at valid_E_idx
            pred_span_scores = (B_scores + E_scores) / 2  # Average of B and E logits
        
        for BECO:
            #dim 0 is the batch and dim 1 is the token idx and dim 2 is the B, E, C or O
            B_scores = logits[valid_B, self.B_class] # Get B logits at valid_B
            E_scores = logits[valid_E - 1, self.E_class] # Get E logits at valid_E-1 (actual end token)
            multi_token_scores = (B_scores + E_scores) / 2  # Average of B and E logits
        '''
        batch, num_spans = span_masks.shape
        device = span_masks.device

        #init the lost_rel_counts, lost_rel_penalty tensors and the rel_labels tensor.
        lost_rel_counts =  torch.zeros(batch, dtype=torch.long, device=device)
        lost_rel_penalty = torch.tensor(0.0, dtype=self.config.torch_precision, device=device)
        #initialize the rel_labels
        rel_labels = self.init_rel_labels(batch, num_spans, device)
 
        #Process each relation and update labels
        #do it in a loop as the number of positive rels are small (typically 0-2 per obs)
        for b in range(batch):
            for head_idx_raw, tail_idx_raw, rel_type in rel_annotations[b]:
                #map the raw head and tail idx to the corresponding span_idx, will be -1 if not found
                head_cand_idx = span_filter_map[b, orig_map[b][head_idx_raw]]
                tail_cand_idx = span_filter_map[b, orig_map[b][tail_idx_raw]]
                #Update rel_labels only if both head and tail span indices are found in span_ids
                #if a head or tail can not be mapped to span_ids it is a lost rel and will be added to the lost_rel_counts
                try:
                    if head_cand_idx != -1 and tail_cand_idx != -1:
                        self.update_rel_labels(rel_labels, b, head_cand_idx, tail_cand_idx, rel_type)
                    
                    #do the lost rel processing as one or both of the head/tail spans are missing
                    ###########################################33
                    else:
                        lost_rel_penalty, lost_rel_counts = self.update_lost_rel_data(all_span_ids, orig_map, token_logits, filter_score_span, head_idx_raw, tail_idx_raw, head_cand_idx, tail_cand_idx, lost_rel_penalty, lost_rel_counts, b)
                    ######################################################3
                except Exception as e:
                    print(e)

        #Flatten tensors for compatibility with downstream processing
        rel_labels = self.flatten_rel_labels(rel_labels)

        return rel_labels, lost_rel_counts, lost_rel_penalty



    def make_rel_ids_and_masks(self, span_masks):
        '''
        so remember the span_masks are derived from the span_masks for the shortlisted spans and remember that the span_masks are 1 for pos cases and selected neg caess from neg sampling
        thus rel_masks here are only 1 if both the head and tail span have mask of 1 and head != tail
        NOTE: depending on run options, if teacher forcing is disabled, then there is no guarantee that all pos cases will be in span_masks
        '''
        batch, num_spans = span_masks.shape
        device = span_masks.device

        #generate rel_ids, using cartesian product straight to (num_spans**2, 2)
        span_indices = torch.arange(num_spans, device=device)    #Generate span indices (0 to num_spans-1 for each item in the batch)
        rel_ids = torch.cartesian_prod(span_indices, span_indices)
        #expand the rel_ids to the batch => to shape (batch, num_spans**2, 2)
        rel_ids = rel_ids.unsqueeze(0).expand(batch, -1, -1)

        #generate rel_masks
        rel_masks = span_masks.unsqueeze(2) & span_masks.unsqueeze(1)
        #Mask out self-relations (i.e., diagonal elements)
        diagonal_rel_mask = torch.eye(num_spans, device=device, dtype=torch.bool)
        rel_masks = rel_masks & ~diagonal_rel_mask.unsqueeze(0)
        #Flatten masks dim 1,2
        rel_masks = rel_masks.view(batch, -1)

        return rel_ids, rel_masks



    def get_rel_tensors(self, all_span_ids, span_masks, rel_annotations, orig_map, span_filter_map, token_logits, filter_score_span):
        """
        Generates relation labels, masks, and IDs for all possible pairs of candidate spans derived from provided span indices.
        
        Args:
            span_masks (torch.Tensor): Boolean tensor indicating valid candidate spans.
            rel_annotations (list of lists of tuples): Each tuple represents a relation as (head, tail, rel_type_string).
            orig_map (list of dicts): Maps annotation span indices to span idx in x['span_ids'] (all poss span ids).
            span_filter_map (torch.Tensor): Maps the span_id in x['span_ids'] (all_span_ids) to the span id in span_ids (pruned span_ids); -1 if not in the pruned span_ids
        
        Returns:
            Tuple of (rel_ids, rel_masks, rel_labels, lost_rel_counts) containing the tensors necessary for further processing.

        called with....
        rel_ids, rel_masks, rel_labels, lost_rel_counts, lost_rel_penalty = self.rel_processor.get_rel_tensors(span_masks,            #the span mask for each selected span  (batch, num_spans)
                                                                                                               rel_annotations,       #the raw relation annotations data.  NOTE: will be None for no labels
                                                                                                               orig_map,              #maps annotation span list id to the dim 1 id in all_span_ids.  NOTE: will be None for no labels  
                                                                                                               span_filter_map)       #maps all_span_ids to current span_ids which will be different after token tagging heads and filtering                    
                            
        """
        #Generate the rel_ids and rel_masks from span_masks
        rel_ids, rel_masks = self.make_rel_ids_and_masks(span_masks)

        #return if we have no labels
        if not self.has_labels:
            return rel_ids, rel_masks, None, None, None

        #get rel_labels if we have labels
        rel_labels, lost_rel_counts, lost_rel_penalty = self.get_rel_labels(rel_annotations, orig_map, span_filter_map, all_span_ids, span_masks, token_logits, filter_score_span) 
        
        return rel_ids, rel_masks, rel_labels, lost_rel_counts, lost_rel_penalty












'''

#################################################################
#testing

def make_rel_ids_and_masks(span_masks):
    """
    so remember the span_masks are derived from the span_masks for the shortlisted spans and remember that the span_masks are 1 for pos cases and selected neg caess from neg sampling
    thus rel_masks here are only 1 if both the head and tail span have mask of 1 and head != tail
    NOTE: depending on run options, if teacher forcing is disabled, then there is no guarantee that all pos cases will be in span_masks
    """
    batch, num_spans = span_masks.shape
    device = span_masks.device

    #generate rel_ids, using cartesian product straight to (num_spans**2, 2)
    span_indices = torch.arange(num_spans, device=device)    #Generate span indices (0 to num_spans-1 for each item in the batch)
    rel_ids = torch.cartesian_prod(span_indices, span_indices)
    #expand the rel_ids to the batch => to shape (batch, num_spans**2, 2)
    rel_ids = rel_ids.unsqueeze(0).expand(batch, -1, -1)

    #generate rel_masks
    rel_masks = span_masks.unsqueeze(2) & span_masks.unsqueeze(1)
    #Mask out self-relations (i.e., diagonal elements)
    diagonal_rel_mask = torch.eye(num_spans, device=device, dtype=torch.bool)
    rel_masks = rel_masks & ~diagonal_rel_mask.unsqueeze(0)
    #Flatten masks dim 1,2
    rel_masks = rel_masks.view(batch, -1)

    return rel_ids, rel_masks
    


# Mock data
batch = 3
num_spans = 3
device = 'cpu'
span_ids = torch.randint(0, 100, (batch, num_spans, 2), device=device)
span_masks = torch.tensor([[True, True, False],[True, True, False],[True, True, False]], dtype=torch.bool, device=device)

# Running the method
rel_ids, rel_masks = make_rel_ids_and_masks(span_masks)

# Printing results for verification
print("span_ids:", span_ids)
print("span_masks:", span_masks)
print("rel_ids shape:", rel_ids.shape)
print("rel_ids:", rel_ids)
print("rel_masks shape:", rel_masks.shape)
print("rel_masks:", rel_masks)
'''


'''
class Config:
    def __init__(self, run_type='train', rel_labels='multilabel', torch_precision=torch.float32, num_rel_types=3, lost_rel_penalty_incr=0.5):
        self.run_type = run_type
        self.rel_labels = rel_labels
        self.torch_precision = torch_precision
        self.num_rel_types = num_rel_types
        self.lost_rel_penalty_incr = lost_rel_penalty_incr
        self.r_to_id = {0:0,1:1,2:2}

# Initialize the RelationProcessor with a multilabel configuration
config = Config()
relation_processor = RelationProcessor(config)

# Define test data for multilabel scenario
span_masks = torch.tensor([[True, True, True, True],
                           [True, True, True, True]], dtype=torch.bool)

# Relation annotations with multilabels (assuming 3 types of relations)
rel_annotations = [
    [(0, 1, 0), (0, 2, 1)],  # Batch 1: Relation between span 0 and 1 of type 0, between 0 and 2 of type 1
    [(0, 2, 2)]              # Batch 2: Relation between span 0 and 2 of type 2
]

orig_map = [
    {0: 4, 1: 2, 2: 3},  # Maps span annotations to all_span_ids indices
    {0: 2, 1: 6, 2: 5}
]

span_filter_map = torch.tensor([
    [-1,-1, 1, 3, 2, 0,-1,-1,-1,-1,-1],  # Mapping from all_span_ids to filtered span_ids
    [-1,-1, 0,-1, 3, 1, 2,-1,-1,-1,-1]
])

# Process the relations for multilabel
rel_ids, rel_masks, rel_labels, lost_rel_counts, lost_rel_penalty = relation_processor.get_rel_tensors(span_masks, rel_annotations, orig_map, span_filter_map)

# Print the outputs to verify the correctness
print("Relation IDs:\n", rel_ids)
print("Relation Masks:\n", rel_masks)
print("Relation Labels (Multilabel):\n", rel_labels)
print("Lost Relation Counts:\n", lost_rel_counts)
print("Lost Relation Penalty:\n", lost_rel_penalty)
'''