from collections import defaultdict
from typing import Tuple, List, Dict, Union

import torch
from torch.utils.data import DataLoader
import random





class DataProcessor(object):
    '''
    This handles the data pre-processing from the incoming json file to the dataloaders
    '''
    def __init__(self, config):
        self.config = config
        #set the has_labels flag based on the self.config.run_type being train (not predict)
        self.has_labels = self.config.run_type == 'train'


    def batch_list_to_dict_converter(self, batch):
        '''
        Converts a batch from a list of dictionaries (one dictionary per observation) into a single dictionary of lists or tensors 
        containing various structured data. This method restructures the batch data for use in further processing or modeling.

        The output dictionary contains:
        - tokens: List of ragged lists containing tokenized sequences.
        - spans: List of ragged lists of tuples representing spans.
        - relations: List of ragged lists of tuples representing relations.
        - orig_map: List of dictionaries mapping original span indices to processed span indices.
        - seq_length: Tensor representing the sequence length of each observation.
        - span_ids: Tensor of span indices, aligned and padded to the batch's maximum sequence length.
        - span_masks: Boolean tensor marking valid spans across the padded length.
        - span_labels: Tensor for labels, supports both unilabel (integers) and multilabel (boolean vectors) formats.

        Parameters:
        - batch (List[Dict]): A list of dictionaries, where each dictionary corresponds to an observation.

        Returns:
        - Dict: A dictionary containing structured data for the batch.

        Notes:
        - Span indices (`span_ids`) and labels (`span_labels`) are aligned to all possible spans defined in `self.config.all_span_ids`.
        - The `span_labels` padding handles both unilabel and multilabel formats by padding with 0 (interpreted as False for multilabel).
        - This function maintains a clear separation of data and control signals, using a mask tensor for validity marking.
        - All properties are initialized in the `dataprocessor.__init__` method, including `self.config.s_to_id`, `self.config.id_to_s`, etc.
        '''
        seq_length  = torch.tensor([x["seq_length"] for x in batch], dtype=torch.long)
        span_ids    = torch.nn.utils.rnn.pad_sequence([obs["span_ids"] for obs in batch], batch_first=True, padding_value=0)
        span_masks  = torch.nn.utils.rnn.pad_sequence([obs["span_masks"] for obs in batch], batch_first=True, padding_value=False)
        tokens      = [obs["tokens"] for obs in batch]
        
        spans, relations, span_labels, orig_map = None, None, None, None
        if self.has_labels:
            spans       = [obs["spans"] for obs in batch]
            relations   = [obs["relations"] for obs in batch]
            span_labels = torch.nn.utils.rnn.pad_sequence([obs["span_labels"] for obs in batch], batch_first=True, padding_value=0)
            orig_map    = [obs['orig_map'] for obs in batch]
        

        # Return a dict of lists
        return dict(
            tokens        = tokens,         #list of ragged lists of strings => the raw word tokenized seq data
            spans         = spans,          #list of ragged lists of tuples => the positive cases for each obs 
            relations     = relations,      #list of ragged lists of tuples => the positive cases for each obs
            orig_map      = orig_map,       #list of dicts for the mapping of the orig span idx to the span_ids dim 1 idx (for pos cases only, is needed in the model for rel tensor generation later)
            seq_length    = seq_length,     #tensor (batch) the length of tokens for each obs
            span_ids      = span_ids,       #tensor (batch, max_seq_len_batch*max_span_width, 2) int => the span_ids truncated to the max_seq_len in the batch* max_span_wdith
            span_masks    = span_masks,     #tensor (batch, max_seq_len_batch*max_span_width) bool => True for valid selected spans, False for rest (padding, invalid spans, unselected neg cases)
            span_labels   = span_labels     #tensor (batch, max_seq_len_batch*max_span_width) int for unilabels.  (batch, max_seq_len_batch*max_span_width, num_span_types) bool for multilabels
        )


    def generate_span_mask_for_obs(self, span_ids, seq_len, max_span_width):
        """
        Generates a mask for a given observation indicating spans that are valid.
        Valid spans are those that do not extend beyond the specified sequence length.

        Args:
        span_ids (torch.Tensor): Tensor of shape (num_spans, 2) containing start and end indices for each span.
        seq_len (int): Length of the sequence, used to determine the validity of each span.

        Returns:
        torch.Tensor: A mask tensor where 1 indicates valid spans, 0 otherwise.
        """
        # Check that end index is greater than start index
        width = span_ids[:, 1] - span_ids[:, 0]
        valid_width = (width > 0) & (width <= max_span_width)
        # Check that spans are within the sequence boundaries
        valid_starts = span_ids[:, 0] >= 0
        valid_ends = span_ids[:, 1] <= seq_len
        # Combine all conditions to determine overall span validity
        return valid_width & valid_starts & valid_ends
        
        #old way of doing it, the same, but a bit less robust
        #return span_ids[:, 1] <= seq_len



    def make_span_labels_unilabels(self, len_span_ids, spans, orig_map):
        """
        This function fills the span_ids aligned span_labels data for the unilabel case
        where each span can have one label only, thus just one integer (0 = negative case, 1+ = positive case).

        Args:
        - len_span_ids (int): The number of span ids.
        - spans (list): List of spans, each with details including the label at the last index.
        - orig_map (list): Mapping from original span indices to the current indices.

        Returns:
        - list: A list of integers where each element corresponds to a label ID or 0 for negative cases.
        """
        span_labels = [0] * len_span_ids
        for i, span in enumerate(spans):
            label = span[-1]
            # Adds the integer ID of the label to the correct span index aligned with span_ids
            if label not in self.config.s_to_id:
                raise ValueError(f'Error. The annotated span type: "{label}" is not in the given span schema, exiting...')
            label_int = self.config.s_to_id[label]
            span_ids_idx = orig_map[i]
            # Check if the annotated data has multiple labels for the unilabel case
            if span_labels[span_ids_idx] != 0:
                #raise ValueError(f'Error. There are multiple labels for span ID {span_ids_idx} and span_labels is set to unilabel, exiting...')
                self.config.logger.write(f'Error. There are multiple labels for span ID {span_ids_idx} and span_labels is set to unilabel', 'warning')
            span_labels[span_ids_idx] = label_int

        return span_labels


    def preprocess_obs(self, obs):
        '''
        NOTE: remember this is working on one obs from the batch (not the entire batch)
        processes the raw obs dict:
        - truncates the tokens
        - simplifies the spans and relations dicts
        - makes the labels tensor for all possible spans
        - sets the spans mask to mask ou tthose invalid spans
        - sets the seq len to the tokens seq len

        NOTE: these params are already set in the self.config params
        self.config.all_span_ids, self.config.s_to_id, self.config.id_to_s, self.config.r_to_id, self.config.id_to_r
        '''
        #Get the max seq length for the batch and truncate to config.max_seq_len if needed
        #####################################################################
        #NOTE: this needs to be as low as possible for model speed, but not so small that it uneccessarily truncates input sequences
        #NOTE: the encoder transformer will later also truncate the encoder specific token sequences (max_enc_seq_len, bert is 512 sw tokens, bigbird is 4096 bigbird tokens)
        #Also remember the mapping of word tokens to encoder tokens will vary between models and input tokens, 
        # but generally the bigbird tokenizer will inflate the word tokens even more than the bert tokenizer, 
        # so this offsets some of the larger context window benefits in bigbird
        max_seq_len = self.config.max_seq_len
        #Truncate based on global self.config.max_seq_len
        tokens = obs['tokens']
        if len(obs['tokens']) > max_seq_len:
            seq_len = max_seq_len
            tokens = tokens[:max_seq_len]
        else:
            seq_len = len(tokens)

        #get all possible spans in seq_len, noting there will be some who's 'end' goes past the seq_len bounds
        #length of span_ids will be seq_len * max_span_width
        span_ids = [x for x in self.config.all_span_ids if x[0] < seq_len]

        #basic processing if we have no labels
        if not self.has_labels:
            span_ids = torch.tensor(span_ids, dtype=torch.long)          #shape => (num_possible_spans)
            span_masks = self.generate_span_mask_for_obs(span_ids, seq_len, self.config.max_span_width)
            spans, relations, span_labels, orig_map = None, None, None, None

        #do label processing and neg sampling as we have labels
        else:
            #make the mapping from original span idx to the idx in the span_ids tensor
            #Map span tuples in span_ids to the index in span_ids for quick lookup
            len_span_ids = len(span_ids)
            span_to_ids_map = {span: idx for idx, span in enumerate(span_ids)}
            #spans = obs['spans']
            #relations = obs['relations']

            #Create a mapping from original span index in annotations to the span index in span_ids (all possible spans in tokens)
            orig_map = {}
            for i, span in enumerate(obs['spans']):
                span_tuple = (span[0], span[1])
                orig_map[i] = span_to_ids_map.get(span_tuple, -1)

            #make the span_labels data which aligns the annotated spans data to the span_ids
            #NOTE: labels will be all negative for the 'predict' case
            #make the unilabel span_labels, when converted to tensor will be shape (num_possible_spans)
            span_labels = self.make_span_labels_unilabels(len_span_ids, obs['spans'], orig_map)
            span_labels = torch.tensor(span_labels, dtype=torch.long)    #shape => (num_possible_spans) for unilabel
            span_ids = torch.tensor(span_ids, dtype=torch.long)          #shape => (num_possible_spans)
            span_masks = self.generate_span_mask_for_obs(span_ids, seq_len, self.config.max_span_width)

        #Return a dictionary with the preprocessed observations
        return dict(
            tokens      = tokens,             #the word tokens of the input seq
            spans       = obs['spans'],       #the simplified list of span tuples [(start, end, type), ...]   NOTE: None for no labels
            relations   = obs['relations'],   #the simplified list of rel tuples [(head, tail, type), ...]    NOTE: None for no labels
            span_ids    = span_ids,           #tensor (seq_len*max_span_width, 2) all possible span (start,end) tuples starting within tokens
            span_labels = span_labels,        #tensor (seq_len*max_span_width) int for unilabel, (seq_len*max_span_width, num span types) bool for multilabel.  The span labels aligning with each element in span_idx.  NOTE: None for no labels
            span_masks  = span_masks,         #tensor (seq_len*max_span_width) the span mask aligning with each element in span_idx, 1 if the span is valid and selected for use   NOTE: 1 for all valid spans and 0 for pad and invalid spans
            orig_map    = orig_map,           #this makes the dict mapping the original span idx in spans to the dim 0 idx in the span_ids tensor here      NOTE: None if no labels
            seq_length  = seq_len,            #length of tokens, a scalar
        )



    def collate_fn(self, batch: List[Dict]) -> Dict:
        """
        Collate a batch of data.
        
        Inputs:
        batch_list => a list of dicts
        
        Output: a dict of lists
        """
        #preprocess the observations
        batch_output = [self.preprocess_obs(obs) for obs in batch]
        #convert the whole batch from a list of dicts to a dict of lists and return
        batch_output = self.batch_list_to_dict_converter(batch_output)
        
        return batch_output



    def create_dataloaders(self, data, **kwargs) -> Dict:
        """
        Create DataLoaders for the dataset with span and relation types extracted from the schema.
        Args:
            data: The dataset to be loaded with train, val, test keys or predict keys for run_type == predict
            **kwargs: Additional arguments passed to the DataLoader.
        Returns:
            one DataLoader per data key: A PyTorch DataLoader instance.
        """
        self.config.logger.write('Making the Dataloaders', 'info')

        #make the loaders    
        if self.config.run_type == 'train':
            loaders = dict(
                train = DataLoader(data['train'], collate_fn=self.collate_fn, batch_size=self.config.train_batch_size, shuffle=self.config.shuffle_train, **kwargs),
                val =   DataLoader(data['val'],   collate_fn=self.collate_fn, batch_size=self.config.eval_batch_size,  shuffle=False, **kwargs),
                test =  DataLoader(data['test'],  collate_fn=self.collate_fn, batch_size=self.config.eval_batch_size,  shuffle=False, **kwargs)
            )

        elif self.config.run_type == 'predict':
            loaders = dict(
                predict =  DataLoader(data['predict'],  collate_fn=self.collate_fn, batch_size=self.config.eval_batch_size,  shuffle=False, **kwargs)
            )
        return loaders








#######################################################################33
#unused code

#this is the code for neg sampling, but for this model, it is not neccessary
#span_masks = self.generate_span_mask_for_obs_w_neg_sampling(span_ids, seq_len, span_labels, self.config.neg_sample_rate, self.config.min_limit)

class blank():
    def calc_neg_samples_count(self, valid_neg_indices, neg_sample_rate, min_limit):
        '''
        Calculate the number of negative samples to select based on the available indices, a specified sample rate, 
        and a minimum limit. This function ensures that the number of selected samples does not exceed the number 
        of available negative indices and respects the defined minimum limit.

        Parameters:
        - valid_neg_indices (list or similar iterable): A collection of indices representing valid negative samples.
        - neg_sample_rate (float): The desired rate of negative sampling as a percentage (e.g., 20 for 20%).
        - min_limit (int): The minimum number of negative samples to be selected, regardless of the sample rate.

        Returns:
        - int: The number of negative samples to select. This number respects the negative sample rate, does not exceed 
        the number of available negatives, and adheres to the minimum limit specified.

        Example:
        - Given 100 valid negative indices, a sample rate of 20%, and a minimum limit of 10, this function will calculate
        an initial sample of 20 (20% of 100). Since 20 is greater than the minimum limit of 10, and less than the total 
        available negatives, the function returns 20.
        '''
        total_negs = len(valid_neg_indices)
        initial_neg_sample = total_negs * (neg_sample_rate / 100)
        negs_after_min_limit = max(initial_neg_sample, min_limit)
        num_negs_to_select = int(min(negs_after_min_limit, total_negs))
        return num_negs_to_select


    def generate_span_mask_for_obs_w_neg_sampling(self, span_ids, seq_len, span_labels=None, neg_sample_rate=None, min_limit=None):
        """
        Generates a single mask for a given observation indicating spans that are both valid and selected. 
        Valid spans are those that do not extend beyond the specified sequence length. The selection includes negative sampling from valid negative spans along with all valid positive spans.

        Args:
        span_ids (torch.Tensor): Tensor of shape (num_spans, 2) containing start and end indices for each span.
        seq_len (int): Length of the sequence, used to determine the validity of each span.
        span_labels (torch.Tensor): Tensor of shape (num_spans) unilabel or (num_spans, num_pos_classes) multilabel containing label values,
                                    where 0 represents negative samples, and >0 represents positive samples.
        neg_sample_rate (float): Fraction of negative samples to randomly select from the valid negative spans.
        min_limit (int): Minimum number of negative samples to retain, if possible.

        Returns:
        torch.Tensor: A mask tensor where 1 indicates selected spans (both negative and positive) from valid spans, 0 otherwise. The selected negative samples are determined through a random selection process adhering to the specified negative sampling rate and minimum limit.

        Process:
        1. Validity Check: Create a mask indicating valid spans based on the end indices not exceeding the sequence length.
        2. Negative Sampling: From valid spans, perform negative sampling on negative labels to determine which negative spans to include.
        3. Final Mask Assembly: Combine the results of negative sampling and the inclusion of all valid positive spans into a single mask indicating which spans are selected for use.
        """
        num_spans, _ = span_ids.shape

        #Create a mask indicating valid spans based on the end indices not exceeding the sequence length
        valid_span_mask = torch.ones(num_spans, dtype=torch.bool)
        valid_span_mask[span_ids[:, 1] > seq_len] = False    #no seq_len-1 here as we have pythonic end indices
        #return if we do not have labels as we do not do any neg samping
        if not self.has_labels:
            return valid_span_mask

        #Determine if span_labels are multilabel and aggregate, binarize in both case for ease of neg sampling
        span_labels_b = (span_labels > 0)

        #make the final span_mask including neg sampling on the valid neg spans and the pos cases
        #############################################
        #Initialize the sample mask as zeros (False), meaning no spans are selected by default
        span_mask = torch.zeros_like(valid_span_mask, dtype=torch.bool)
        #Find indices where spans are valid
        valid_indices = torch.nonzero(valid_span_mask, as_tuple=True)[0]
        valid_labels = span_labels_b[valid_indices]
        #Separate indices for negative and positive valid labels
        valid_neg_indices = torch.nonzero(valid_labels == False, as_tuple=True)[0]
        valid_pos_indices = torch.nonzero(valid_labels == True, as_tuple=True)[0]
        ##########################################
        #do the neg sampling adhering to the sample rate and the min limit
        num_negs_to_select = self.calc_neg_samples_count(valid_neg_indices, neg_sample_rate, min_limit)
        selected_neg_indices = valid_neg_indices[torch.randperm(len(valid_neg_indices))[:num_negs_to_select]]
        ##########################################
        #Map selected negative indices back to the original span_labels tensor indices
        selected_neg_indices = valid_indices[selected_neg_indices]
        valid_pos_indices = valid_indices[valid_pos_indices]
        #set the selected neg_ind and valid_pos_ind to True in the sample_mask
        span_mask[selected_neg_indices] = True
        span_mask[valid_pos_indices] = True

        return span_mask
