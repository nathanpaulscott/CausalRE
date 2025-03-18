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



    def init_rel_labels(self):
        '''
        makes the blank rel_labels tensors
        '''
        if self.config.rel_labels == 'unilabel':
            return torch.full((self.batch, self.top_k_spans, self.top_k_spans), -1, dtype=torch.long, device=self.device)
        elif self.config.rel_labels == 'multilabel':
            return torch.zeros((self.batch, self.top_k_spans, self.top_k_spans, self.config.num_rel_types), dtype=torch.bool, device=self.device)


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
        unilabel => (batch, top_k_spans, top_k_spans) => (batch, top_k_spans**2)
        multilabel => (batch, top_k_spans, top_k_spans, num rel types) => (batch, top_k_spans**2, num rel types)
        '''
        if self.config.rel_labels == 'unilabel':
            return rel_labels.view(self.batch, -1)
        elif self.config.rel_labels == 'multilabel':
            return rel_labels.view(self.batch, -1, self.config.num_rel_types)


    def get_rel_labels(self, raw_rels, orig_map, span_filter_map):  #, x, t, sik, swsids, fss, sm, sl):
        #init the lost_rel_counts, lost_rel_penalty tensors and the rel_labels tensor.
        lost_rel_counts =  torch.zeros(self.batch, dtype=torch.long, device=self.device)
        #NOTE: the lost_rel_penalty is for calculting the penalty loss for lost rels, so it needs to be shape (1) float and requires_grad
        lost_rel_penalty = torch.tensor(0.0, dtype=self.config.torch_precision, device=self.device, requires_grad=True)
        #initialize the rel_labels
        rel_labels = self.init_rel_labels()

        #Process each relation and update labels
        #do it in a loop as the number of positive rels are small (typically 0-2 per obs)
        for i in range(self.batch):
            for head_idx_raw, tail_idx_raw, rel_type in raw_rels[i]:
                #map the raw head and tail idx to the corresponding span_idx, will be -1 if not found
                head_cand_idx = span_filter_map[i, orig_map[i][head_idx_raw]]
                tail_cand_idx = span_filter_map[i, orig_map[i][tail_idx_raw]]
                #Update rel_labels only if both head and tail span indices are found in span_ids
                #if a head or tail can not be mapped to span_ids it is a lost rel and will be added to the lost_rel_counts
                try:
                    if head_cand_idx != -1 and tail_cand_idx != -1:
                        self.update_rel_labels(rel_labels, i, head_cand_idx, tail_cand_idx, rel_type)
                    else:    #here we have a lost relation case caused by one or 2 missing spans in top_k_spans
                        #self.config.logger.write('lost rel')
                        #for debugging
                        '''
                        print(f'\n!!lost rel!!: head_idx_raw: {head_idx_raw}, tail_idx_raw: {tail_idx_raw}, span_id_head: {orig_map[i][head_idx_raw]}, span_id_tail: {orig_map[i][tail_idx_raw]}, head_cand_idx: {head_cand_idx}, tail_cand_idx: {tail_cand_idx}')
                        print(x['tokens'])
                        print(x['spans'])
                        print(x['relations'])
                        print(f'top_k_spans: {t}')
                        print(f'span_idx_to_keep: {sik}')
                        print(orig_map[i])
                        print(span_filter_map[i]) 
                        for id,  (sid, swsid, score, span_idx, mask, label) in enumerate(zip(x['span_ids'][i], swsids[i], fss[i], span_filter_map[i], sm[i], sl[i])):
                            msg = f'{id}: {x["tokens"][i][sid[0]] if sid[0] < len(x["tokens"][i]) else "---"}:{x["tokens"][i][sid[1]-1] if sid[1]-1 < len(x["tokens"][i]) else "---"} => {sid.tolist()}, {swsid.tolist()}, {round(score.item(),4)}, {span_idx.item()}, {mask.item()}, {label.item()}'
                            if id == orig_map[i][head_idx_raw]:
                                msg += ' *head*'
                            elif id == orig_map[i][tail_idx_raw]:
                                msg += ' *tail*'
                            print(msg)

                        exit()
                        '''
                    
                        #increment the lost rel counter for this batch
                        lost_rel_counts[i] += 1
                        #add the lost_rel_penalty
                        lost_rel_penalty = lost_rel_penalty + self.config.lost_rel_penalty_incr
                        if head_cand_idx == -1 and tail_cand_idx == -1:   #double the penalty if both spans are missing
                            #add an extra penalty if both spans are missing causing the lost rel causing the lost rel
                            lost_rel_penalty = lost_rel_penalty + self.config.lost_rel_penalty_incr
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
        NOTE: that we are NOT doing neg sampling here, I am not actually sure how I would even include that as we now can have lost rels (pos rels from labels that never maade it to the ids as they were misclassified by spans)
        I think you would include it here in the rel_masks, you basically would look for 100 or the largest available number of rels where both head and tail are pos case spans, i.e. spans.  Thus if there are only a few
        actual spans, then the number of neg cases is not large.....
        This is not so important if we are using the pruning stage to get cand spans and can rels!!!
        '''
        #generate rel_ids, using cartesian product straight to (top_k_spans**2, 2)
        top_k_span_indices = torch.arange(self.top_k_spans, device=self.device)    #Generate span indices (0 to top_k_spans-1 for each item in the batch)
        rel_ids = torch.cartesian_prod(top_k_span_indices, top_k_span_indices)
        #expand the rel_ids to the batch => to shape (batch, top_k_spans**2, 2)
        rel_ids = rel_ids.unsqueeze(0).expand(self.batch, -1, -1)

        #generate rel_masks
        rel_masks = span_masks.unsqueeze(2) & span_masks.unsqueeze(1)
        #Mask out self-relations (i.e., diagonal elements)
        diagonal_rel_mask = torch.eye(self.top_k_spans, device=self.device, dtype=torch.bool)
        rel_masks = rel_masks & ~diagonal_rel_mask.unsqueeze(0)
        #Flatten masks dim 1,2
        rel_masks = rel_masks.view(self.batch, -1)

        return rel_ids, rel_masks



    def get_rel_tensors(self, span_masks, raw_rels, orig_map, span_filter_map):    #), x, t, sik, swsids, fss, sm, sl):
        """
        Generates relation labels, masks, and IDs for all possible pairs of candidate spans derived from provided span indices.
        This function is crucial for dynamic and efficient relation extraction, adapting to both unilabel and multilabel scenarios and ensuring that lost relations are tracked.

        Args:
            span_ids (torch.Tensor): Tensor containing indices of candidate spans.
            span_masks (torch.Tensor): Boolean tensor indicating valid candidate spans.
            raw_rels (list of lists of tuples): Each tuple represents a relation as (head, tail, rel_type_string).
            orig_map (list of dicts): Maps original span indices to candidate span indices for each batch.
            span_filter_map (torch.Tensor): Maps the span_id idx to span_ids idx; -1 if not in candidate spans.
        
        Returns:
            Tuple of (rel_ids, rel_masks, rel_labels, lost_rel_counts) containing the tensors necessary for further processing.
        """
        self.batch, self.top_k_spans = span_masks.shape
        self.device = span_masks.device

        #Generate the rel_ids and rel_masks from the span_ids and span_masks
        rel_ids, rel_masks = self.make_rel_ids_and_masks(span_masks)

        #return if we have no labels
        if not self.has_labels:
            return rel_ids, rel_masks, None, None

        #get rel_labels if we have labels
        rel_labels, lost_rel_counts, lost_rel_penalty = self.get_rel_labels(raw_rels, orig_map, span_filter_map)   #, x, t, sik, swsids, fss, sm, sl)
        
        return rel_ids, rel_masks, rel_labels, lost_rel_counts, lost_rel_penalty












'''

#################################################################
#testing

def make_rel_ids_and_masks(span_ids, span_masks, batch, top_k_spans, device):
    #generate rel_ids, using cartesian product stright to (batch, top_k_spans**2, 2)
    indices = torch.arange(top_k_spans, device=device)    #Generate span indices (0 to top_k_spans-1 for each item in the batch)
    rel_ids = torch.cartesian_prod(indices, indices)
    rel_ids = rel_ids.unsqueeze(0).expand(batch, -1, -1)

    #generate rel_masks
    rel_masks = span_masks.unsqueeze(2) & span_masks.unsqueeze(1)
    #Mask out self-relations (i.e., diagonal elements)
    diagonal_mask = torch.eye(top_k_spans, device=device, dtype=torch.bool).unsqueeze(0)
    rel_masks = rel_masks & ~diagonal_mask
    #Flatten dim 1,2 of masks for compatibility
    rel_masks = rel_masks.view(batch, -1)

    return rel_ids, rel_masks


# Mock data
batch = 3
top_k_spans = 3
device = 'cpu'
span_ids = torch.randint(0, 100, (batch, top_k_spans, 2), device=device)
span_masks = torch.tensor([[True, True, False],[True, True, False],[True, True, False]], dtype=torch.bool, device=device)

# Running the method
rel_ids, rel_masks = make_rel_ids_and_masks(span_ids, span_masks, batch, top_k_spans, device)

# Printing results for verification
print("span_ids:", span_ids)
print("span_masks:", span_masks)
print("rel_ids shape:", rel_ids.shape)
print("rel_ids:", rel_ids)
print("rel_masks shape:", rel_masks.shape)
print("rel_masks:", rel_masks)
'''