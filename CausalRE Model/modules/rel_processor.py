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
        self.has_labels = config.run_type == 'train'



    def dynamic_top_k_rels(self, rel_scores, config):
        """
        Dynamically determines the number of top relationships to include in the graph based on a percentile threshold of relationship scores.

        This function calculates the threshold score at a specified percentile of the relationship scores provided. It then counts how many relationships exceed this threshold and limits the count based on the maximum number allowed by the configuration. The function ensures that the number of relationships considered does not exceed the actual number of relationships available.

        Args:
            rel_scores (torch.Tensor): A tensor containing the scores of relationships, where higher scores indicate a stronger likelihood of the relationship being significant. These scores should range from -inf to +inf.
            config (ConfigClass): An object containing configuration parameters, including:
                rel_score_percentile (float): The percentile (between 0 and 1) used to determine the threshold score for including relationships. For example, 0.95 means the top 5% of scores are considered.
                max_top_k_rels (int): The maximum number of relationships to include, providing an upper bound to prevent excessive computation.

        Returns:
            int: The number of relationships to include, which is the minimum of the number of relationships exceeding the score threshold, the maximum number allowed, and the total number of relationships available.

        Example:
            >>> rel_scores = torch.tensor([0.1, 0.4, 0.35, 0.8, 0.95])
            >>> config = ConfigClass(rel_score_percentile=0.8, max_top_k_rels=3)
            >>> dynamic_top_k_rels(rel_scores, config)
            2  # Only two scores exceed the 80th percentile threshold in this example.
        """
        rel_score_thd = torch.quantile(rel_scores, config.rel_score_percentile)
        valid_rels_mask = rel_scores >= rel_score_thd
        valid_rels_count = valid_rels_mask.sum().item()
        top_k_rels = min(valid_rels_count, config.max_top_k_rels, rel_scores.shape[1])
        return top_k_rels



    def init_rel_labels(self, batch, top_k_spans, device):
        if self.config.rel_labels == 'unilabel':
            return torch.full((batch, top_k_spans, top_k_spans), -1, dtype=torch.int32, device=device)
        elif self.config.rel_labels == 'multilabel':
            return torch.zeros((batch, top_k_spans, top_k_spans, self.config.num_rel_types), dtype=torch.bool, device=device)


    def update_rel_labels(self, rel_labels, batch_idx, head_cand_idx, tail_cand_idx, rel_type):
        r_label_id = self.config.r_to_id[rel_type]
        if self.config.rel_labels == 'unilabel':
            rel_labels[batch_idx, head_cand_idx, tail_cand_idx] = r_label_id
        elif self.config.rel_labels == 'multilabel':
            #convert rel label id to a position in the multilabel binary label vector
            #NOTE: the r_label_id for the multilabel case will be the position in the vector as idx 0 is the first pos class
            rel_labels[batch_idx, head_cand_idx, tail_cand_idx, r_label_id] = True


    def flatten_rel_labels(self, batch, rel_labels):
        if self.config.rel_labels == 'unilabel':
            return rel_labels.view(batch, -1)
        elif self.config.rel_labels == 'multilabel':
            return rel_labels.view(batch, -1, self.config.num_rel_types)


    def get_rel_labels(self, raw_rels, orig_map, span_to_cand_span_map, batch, top_k_spans, device):
        #init the lost_rel_counts tensor and the rel_labels tensor
        lost_rel_counts = torch.zeros(batch, dtype=torch.int32, device=device)
        rel_labels = self.init_rel_labels(batch, top_k_spans, device)
        #Process each relation and update labels
        #do it in a loop as the number of positive rels are small (typically 0-2 per obs)
        for i in range(batch):
            for head_idx_raw, tail_idx_raw, rel_type in raw_rels[i]:
                #map the raw head and tail idx to the corresponding cand_span_idx, will be -1 if not found
                head_cand_idx = span_to_cand_span_map[i, orig_map[i][head_idx_raw]]
                tail_cand_idx = span_to_cand_span_map[i, orig_map[i][tail_idx_raw]]
                #Update rel_labels only if both head and tail span indices are found in cand_span_ids
                #if a head or tail can not be mapped to cand_span_ids it is a lost rel and will be addedto the lost_rel_counts
                if head_cand_idx != -1 and tail_cand_idx != -1:
                    self.update_rel_labels(rel_labels, i, head_cand_idx, tail_cand_idx, rel_type)
                else:
                    #this adds one to the count for each annotated rel that has no rel reps to classify
                    #so it is not one for each lost head/tail pair, it is one for each lost head/tail/type triple
                    #this is the best way to penalize the model via loss for lost rels triples. 
                    #NOTE: I have the rel_id, but I am not storing it, this may need to be done at a later point (maybe not)
                    lost_rel_counts[i] += 1

        #Flatten tensors for compatibility with downstream processing
        rel_labels = self.flatten_rel_labels(batch, rel_labels)

        return rel_labels, lost_rel_counts


    def make_rel_ids_and_masks(self, cand_span_ids, cand_span_masks, batch, top_k_spans, device):
        #generate rel_ids
        rel_ids = torch.stack((
            cand_span_ids.unsqueeze(2).repeat(1, 1, top_k_spans),
            cand_span_ids.unsqueeze(1).repeat(1, top_k_spans, 1)
        ), dim=3)

        #generate rel_masks
        rel_masks = cand_span_masks.unsqueeze(2) & cand_span_masks.unsqueeze(1)
        #Mask out self-relations (i.e., diagonal elements)
        diagonal_mask = torch.eye(top_k_spans, device=device, dtype=torch.bool).unsqueeze(0)
        rel_masks = rel_masks & ~diagonal_mask

        #Flatten tensors for compatibility with downstream processing
        rel_ids = rel_ids.view(batch, -1, 2)
        rel_masks = rel_masks.view(batch, -1)

        return rel_ids, rel_masks



    def get_cand_rel_tensors(self, cand_span_ids, cand_span_masks, raw_rels, orig_map, span_to_cand_span_map):
        """
        Generates relation labels, masks, and IDs for all possible pairs of candidate spans derived from provided span indices.
        This function is crucial for dynamic and efficient relation extraction, adapting to both unilabel and multilabel scenarios and ensuring that lost relations are tracked.

        Args:
            cand_span_ids (torch.Tensor): Tensor containing indices of candidate spans.
            cand_span_masks (torch.Tensor): Boolean tensor indicating valid candidate spans.
            raw_rels (list of lists of tuples): Each tuple represents a relation as (head, tail, rel_type_string).
            orig_map (list of dicts): Maps original span indices to candidate span indices for each batch.
            span_to_cand_span_map (torch.Tensor): Maps the span_id idx to cand_span_ids idx; -1 if not in candidate spans.
        
        Returns:
            Tuple of (rel_ids, rel_masks, rel_labels, lost_rel_counts) containing the tensors necessary for further processing.
        """
        batch, top_k_spans = cand_span_masks.shape
        device = cand_span_masks.device

        #Generate the rel_ids and rel_masks from the cand_span_ids and cand_span_masks
        rel_ids, rel_masks = self.make_rel_ids_and_masks(cand_span_ids, cand_span_masks, batch, top_k_spans, device)

        #return if we have no labels
        if not self.has_labels:
            return rel_ids, rel_masks, None, None

        #get rel_labels if we have labels
        rel_labels, lost_rel_counts = self.get_rel_labels(raw_rels, orig_map, span_to_cand_span_map, batch, top_k_spans, device)
        
        return rel_ids, rel_masks, rel_labels, lost_rel_counts

