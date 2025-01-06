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
        seq_length = torch.tensor([x["seq_length"] for x in batch], dtype=torch.long)
        span_ids = torch.nn.utils.rnn.pad_sequence([obs["span_ids"] for obs in batch], batch_first=True, padding_value=0)
        span_masks = torch.nn.utils.rnn.pad_sequence([obs["span_mask"] for obs in batch], batch_first=True, padding_value=False)
        span_labels = torch.nn.utils.rnn.pad_sequence([obs["span_label"] for obs in batch], batch_first=True, padding_value=0)
        tokens = [obs["tokens"] for obs in batch]
        spans = [obs["spans"] for obs in batch]
        relations = [obs["relations"] for obs in batch]
        orig_map = [obs['orig_map'] for obs in batch]
        # Return a dict of lists
        return dict(
            tokens        = tokens,         #list of ragged lists of strings => the raw word tokenized seq data
            spans         = spans,          #list of ragged lists of tuples => the positive cases for each obs 
            relations     = relations,      #list of ragged lists of tuples => the positive cases for each obs
            orig_map      = orig_map,       #list of dicts for the mapping of the orig span idx to the span_ids dim 1 idx (for pos cases only, is needed in the model for rel tensor generation later)
            seq_length    = seq_length,     #tensor (batch) the length of tokens for each obs
            span_ids      = span_ids,       #tensor (batch, max_seq_len_batch*max_span_width, 2) int => the span_ids truncated to the max_seq_len in the batch* max_span_wdith
            span_masks    = span_masks,     #tensor (batch, max_seq_len_batch*max_span_width) bool => True for valid selected spans, False for rest (padding, invalid spans, unselected neg cases)
            span_labels   = span_labels,    #tensor (batch, max_seq_len_batch*max_span_width) int for unilabels.  (batch, max_seq_len_batch*max_span_width, num_span_types) bool for multilabels
        )



    def calc_neg_samples_count(valid_neg_indices, neg_sample_rate, min_limit):
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


    def generate_span_mask_for_obs(self, span_labels, span_ids, seq_len, neg_sample_rate, min_limit):
        """
        Generates a single mask for a given observation indicating spans that are both valid and selected. 
        Valid spans are those that do not extend beyond the specified sequence length. The selection includes negative sampling from valid negative spans along with all valid positive spans.

        Args:
        span_labels (torch.Tensor): Tensor of shape (num_spans) unilabel or (num_spans, num_pos_classes) multilabel containing label values,
                                    where 0 represents negative samples, and >0 represents positive samples.
        span_ids (torch.Tensor): Tensor of shape (num_spans, 2) containing start and end indices for each span.
        seq_len (int): Length of the sequence, used to determine the validity of each span.
        neg_sample_rate (float): Fraction of negative samples to randomly select from the valid negative spans.
        min_limit (int): Minimum number of negative samples to retain, if possible.

        Returns:
        torch.Tensor: A mask tensor where 1 indicates selected spans (both negative and positive) from valid spans, 0 otherwise. The selected negative samples are determined through a random selection process adhering to the specified negative sampling rate and minimum limit.

        Process:
        1. Validity Check: Create a mask indicating valid spans based on the end indices not exceeding the sequence length.
        2. Negative Sampling: From valid spans, perform negative sampling on negative labels to determine which negative spans to include.
        3. Final Mask Assembly: Combine the results of negative sampling and the inclusion of all valid positive spans into a single mask indicating which spans are selected for use.
        """
        #Determine if span_labels are multilabel and aggregate to unilabel if so for ease of neg sampling
        if self.config.span_labels == 'multilabel':
            span_labels = (torch.sum(span_labels, dim=1) > 0)
        else:  # Unilabel case
            span_labels = (span_labels > 0)

        # Create a mask indicating valid spans based on the end indices not exceeding the sequence length
        valid_span_mask = torch.ones(span_labels.shape[0], dtype=torch.bool)
        valid_span_mask[span_ids[:, 1] > seq_len] = False    #no seq_len-1 here as we have pythonic end indices

        #make the final span_mask including neg sampling on the valid neg spans and the pos cases
        #############################################
        #Initialize the sample mask as zeros (False), meaning no spans are selected by default
        span_mask = torch.zeros_like(valid_span_mask, dtype=torch.bool)
        #Find indices where spans are valid
        valid_indices = torch.nonzero(valid_span_mask, as_tuple=True)[0]
        valid_labels = span_labels[valid_indices]
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
                raise ValueError(f'Error. There are multiple labels for span ID {span_ids_idx} and span_labels is set to unilabel, exiting...')
            span_labels[span_ids_idx] = label_int

        return span_labels




    def make_span_labels_multilabels(self, len_span_ids, num_pos_classes, spans, orig_map):
        '''
        This function fills the span_ids aligned span_labels data for the multilabel case,
        where each span has a binary vector for labels. The vector length equals num_pos_classes,
        which does not include the none_span (index 0).

        Args:
        - len_span_ids (int): The number of span ids.
        - num_pos_classes (int): The number of positive classes, excluding the none type.
        - spans (list): List of spans, each with details including the label at the last index.
        - orig_map (list): Mapping from the original span indices to the current indices.

        Returns:
        - list: A list of lists, where each inner list is a binary vector indicating the presence of any of the defined positive labels.
        '''
        span_labels = [[False]*num_pos_classes for x in range(len_span_ids)]
        for i, span in enumerate(spans):
            label = span[-1]
            #check the label is in the schema
            if label not in self.config.s_to_id:
                raise Exception(f'Error. The span type: "{label}" is not in the given span schema, exiting...')
            #Get the label index adjusted for zero-based indexing in Python
            label_vector_idx = self.config.s_to_id[label]    #no need to adjust the idx as for the multilabel case only pos types are in self.config.span_types 
            #Update the binary vector for that span
            span_labels[orig_map[i]][label_vector_idx] = True  #Set to True where label occurs
        
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
        #get teh neg sampling config params
        neg_sample_rate = self.config.neg_sample_rate, 
        min_neg_sample_limit = self.config.min_neg_sample_limit
        
        #Get the maximum length for the word tokens and truncate
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
        len_span_ids = len(span_ids)
        #make the mapping from original span idx to the idx in the span_ids tensor
        #Map span tuples in span_ids to the index in span_ids for quick lookup
        span_to_ids_map = {span: idx for idx, span in enumerate(span_ids)}

        #simplify the spans and rels to list of tuples
        if self.config.run_type == 'predict':
            #only need this for predict and everything else just falls into palce,
            #NOTE: we disable neg samping by setting the rate to 100%
            spans, rels, neg_sample_rate = [], [], 100
        else:
            spans = [(x['start'], x['end'], x['type']) for x in obs['spans']]
            rels = [(x['head'], x['tail'], x['type']) for x in obs['relations']]

        #Create a mapping from original span index in annotations to the span index in span_ids (all possible spans in tokens)
        orig_map = {}
        for i, span in enumerate(spans):
            span_tuple = (span[0], span[1])
            orig_map[i] = span_to_ids_map[span_tuple]

        #make the span_labels data which aligns the annotated spans data to the span_ids
        if self.config.span_labels == 'unilabel':
            #make the unilabel span_labels, when converted to tensor will be shape (num_possible_spans)
            span_labels = self.make_span_labels_unilabels(len_span_ids, spans, orig_map)
            #move to tensors
            span_labels = torch.tensor(span_labels, dtype=torch.long)    #shape => (num_possible_spans) for unilabel

        elif self.config.span_labels == 'multilabel':
            #make the multilabel span_labels, when converted to tensor will be shape (num_possible_spans, num_span_types)
            span_labels = self.make_span_labels_multilabels(len_span_ids, self.config.num_span_types, spans, orig_map)
            #move to tensors
            span_labels = torch.tensor(span_labels, dtype=torch.bool)    #shape => (num_possible_spans, num_span_types) for multilabel

        #move span_ids to tensors
        span_ids = torch.tensor(span_ids, dtype=torch.long)          #shape => (num_possible_spans)
        
        #do neg sampling to get the span_mask (valid pos spans + valid selected neg cases)
        #NOTE: it automatically handles both unilabel and multilabel cases
        span_mask = self.generate_span_mask_for_obs(span_labels, span_ids, seq_len, neg_sample_rate, min_neg_sample_limit)

        # Return a dictionary with the preprocessed observations
        return dict(
            tokens      = tokens,             #the word tokens of the input seq
            spans       = spans,              #the simplified list of span tuples [(start, end, type), ...]   NOTE: [] for no labels
            relations   = rels,               #the simplified list of rel tuples [(head, tail, type), ...]    NOTE: [] for no labels
            span_ids    = span_ids,           #tensor (seq_len*max_span_width, 2) all possible span (start,end) tuples starting within tokens
            span_label  = span_labels,        #tensor (seq_len*max_span_width) int for unilabel, (seq_len*max_span_width, num span types) bool for multilabel.  The span labels aligning with each element in span_idx.
            span_mask   = span_mask,          #tensor (seq_len*max_span_width) the span mask aligning with each element in span_idx, 1 if the span is valid and selected for use   NOTE: 1 for all valid spans and 0 for pad and invalid spans
            orig_map    = orig_map,           #this makes the dict mapping the original span idx in spans to the dim 0 idx in the span_ids tensor here      NOTE: {} if no labels
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




################################################################################################
################################################################################################
################################################################################################
################################################################################################
################################################################################################



class PromptProcessor():
    '''
    Handles the adding and removal of the prompt to the tokens
    '''
    def __init__(self, config):
        self.config = config


    def make_prompt(self, types, type_token):
        prompt = []
        for type in types:
            prompt.append(type_token)
            prompt.append(type)
        prompt.append(self.config.sep_token)
        return prompt



    def add_prompt_to_tokens(self, x):
        '''
        This adds in prompt tokens for each span and rel type to the tokens for each obs (including the none type for the unilabel case)
        x is the batch, x['tokens'] is a list of lists, the outer list is for the observations, the inner list is for tokens in the sequence
        eg.
        tokens_w_prompt = ['<<S>>', 'span_type1', '<<S>>', 'span_type2', '<<SEP>>', '<<R>>','rel_type1','<<R>>','rel_type2', <<SEP>>, normal input tokens...]
        
        NOTE: 
        for unilabels => self.config.span/rel_types CONTAIN the none type at idx 0
        for multilabels => self.config.span/rel_types DO NOT contain the none type, idx 0 is the first pos type
        '''
        #this puts all the unique entity classes in teh obs into a list of strings ['<<S>>', span_type1, '<<S>>', span_type2,.... '<<SEP>>']
        span_prompt = self.make_prompt(self.config.span_types, self.config.s_token)
        span_prompt_len = len(span_prompt)
        #this puts all the unique relation classes in the obs into a list of strings ['<<R>>', relation_class1, '<<R>>', relation_class2,.... '<<SEP>>']
        rel_prompt = self.make_prompt(self.config.rel_types, self.config.r_token)
        #concats the span prompt with the rel prompt
        prompt = span_prompt + rel_prompt
        prompt_len = len(prompt)
        #add to the tokens
        prompt_w_tokens = [prompt + tokens for tokens in x["tokens"]]
        len_prompt_w_tokens = x["seq_length"] + prompt_len
        
        return dict(
            prompt_x        = prompt_w_tokens, 
            len_prompt_x    = len_prompt_w_tokens,
            prompt_len      = prompt_len,        #length of total prompt in word tokens + 2 for the 2 ending <<SEP>> tokens (one after spans and one after rels)
            span_prompt_len = span_prompt_len    #length of span prompt in word tokens + 1 for the ending <<SEP>> token
            )

 

    def split_embeddings(self, embeddings, masks, prompt_len, span_prompt_len, w2sw_map=None, input_ids=None, word_tokens=None, tokenizer=None):
        '''
        Splits embeddings into token representations, span type representations and relation type representations by separating out prompt tokens from non-prompt tokens.

        Args:
            embeddings: Tensor of shape (batch, seq_len, hidden) containing token embeddings with [CLS], [SEP] removed
            masks: Tensor of shape (batch, seq_len) containing attention masks with [CLS], [SEP] removed
            prompt_len: Integer length of full prompt in word tokens
            span_prompt_len: Integer length of span prompt portion in word tokens 
            w2sw_map: Optional list of dicts mapping word indices to subword indices, adjusted for [CLS], [SEP] removal
            input_ids: Optional tensor of input token ids with [CLS], [SEP] removed
            word_tokens: Optional list of original word tokens including prompt
            tokenizer: Optional tokenizer object for debugging

        Returns:
            Dictionary containing:
            - token_reps: Tensor (batch, max_seq_len, hidden) of non-prompt token representations
            - token_masks: Tensor (batch, max_seq_len) of attention masks for non-prompt tokens
            - span_type_reps: Tensor (batch, num_span_types, hidden) of span type representations
            - rel_type_reps: Tensor (batch, num_rel_types, hidden) of relation type representations  
            - w2sw_map: List of dicts with adjusted word-to-subword mappings to account for prompt removal if no pooling used

        Notes:
            - All inputs should have [CLS], [SEP] tokens/positions already removed
            - For subtoken_pooling='none': Works with subword tokens, extracts <<S>>/<<R>> embeddings
            - For subtoken_pooling='mean'/'maxpool': Works with word tokens, extracts prompt token embeddings
            - w2sw_map is adjusted to account for prompt removal in the no pooling case
        '''
        token_reps      = embeddings
        token_masks     = masks
        span_type_reps  = None
        rel_type_reps   = None
        w2sw_map_new    = None

        if self.config.subtoken_pooling == 'none':
            #For no pooling case, work with subword indices, this will be the HF model source
            #extract the prompt and non-prompt embeddings if the prompt is there
            if prompt_len > 0:  #no pooling with prompts
                #Get first batch mapping (prompt structure should be same for all batches)
                #NOTE: both are adjusted to account for removing the CLS token above
                first_map = w2sw_map[0]
                first_input_ids = input_ids[0]   
                #Get the len of the span and total prompt in subtokens, including the <<SEP>> tokens
                span_prompt_len_subtoken = max(first_map[span_prompt_len - 1]) + 1 if span_prompt_len > 0 else 0
                total_prompt_len_subtoken = max(first_map[prompt_len - 1]) + 1 if prompt_len > 0 else 0

                #Get span type representations, actually the <<S>> token embedding preceding each group of span_type subword tokens
                #NOTE: the indices are the same for all obs
                #subtract 1 from the last of the range as the last idx is <<SEP>>
                s_token_indices = [i for i in range(0, span_prompt_len_subtoken-1) if first_input_ids[i] == self.config.s_token_id]
                span_type_reps = token_reps[:, s_token_indices, :]
                
                #Get rel type representations, actually the <<R>> token embedding preceding each group of rel_type subword tokens
                #NOTE: the indices are the same for all obs
                #subtract 1 from the last of the range as the last idx is <<SEP>>
                r_token_indices = [i for i in range(span_prompt_len_subtoken, total_prompt_len_subtoken-1) if first_input_ids[i] == self.config.r_token_id]
                rel_type_reps = token_reps[:, r_token_indices, :]

                #Split out the embeddings for the non prompt tokens
                token_indices = list(range(total_prompt_len_subtoken, token_reps.shape[1]))
                token_reps = token_reps[:, token_indices, :]
                token_masks = token_masks[:, token_indices]

                #adjust w2sw_map to account for prompt word tokens removal and prompt subtokens removal
                def mod_sw_mapping(subtokens, sw_token_offset):
                    return [x - sw_token_offset for x in subtokens if x >= sw_token_offset]
                
                def fix_mapping(mapping, word_token_offset, sw_token_offset):
                    shifted = {k-word_token_offset: mod_sw_mapping(v, sw_token_offset) for k, v in mapping.items()}
                    return {k: v for k, v in shifted.items() if v}

                w_offset = prompt_len    #in word tokens
                sw_offset = total_prompt_len_subtoken  #in subtokens
                w2sw_map_new = [fix_mapping(mapping, w_offset, sw_offset) for mapping in w2sw_map]

                '''
                #for debugging
                print(span_prompt_len)
                print(prompt_len)
                print(span_prompt_len_subtoken)
                print(total_prompt_len_subtoken)
                #to see the actual subword tokens for debugging
                #adjust incides as input_ids has cls/sep subtokens
                print(tokenizer.convert_ids_to_tokens(first_input_ids[token_indices]))
                print(tokenizer.convert_ids_to_tokens(first_input_ids[s_token_indices]))
                print(tokenizer.convert_ids_to_tokens(first_input_ids[r_token_indices]))
                subtokens = tokenizer.convert_ids_to_tokens(first_input_ids)
                no_prompt_subtokens = tokenizer.convert_ids_to_tokens(first_input_ids[token_indices])
                print("\nSubword tokens sequence:")
                print([(i, x) for i,x in enumerate(subtokens)])

                # Get actual subword tokens for first sequence using input_ids
                print("\nPrompt tokens with word positions:")
                for word_idx, subword_positions in first_map.items():
                    subtoken_list = [(subtokens[pos], pos) for pos in subword_positions]
                    print(f"Word position {word_idx}: {subtoken_list}")


                # Get actual subword tokens for first sequence using input_ids
                print("\n map with promtp removed:")
                for word_idx, subword_positions in w2sw_map_new[0].items():
                    subtoken_list = [(no_prompt_subtokens[pos], pos) for pos in subword_positions]
                    print(f"Word position {word_idx}: {subtoken_list}")


                '''

        else:     #pooling case
            #all embeddings are mapped 1:1 with the pre transformer word tokens
            #there are no special transformer tokens [CLS], [SEP] as they have either been removed (HF) or flair removed them itself
            #get the span_type reps
            s_token_indices = list(range(0, span_prompt_len-1, 2))
            span_type_reps = token_reps[:, s_token_indices, :]
            
            #get the rel_type reps
            r_token_indices = list(range(span_prompt_len, prompt_len-1, 2))
            rel_type_reps = token_reps[:, r_token_indices, :]

            #finally get the token reps and mask
            token_indices = list(range(prompt_len, token_reps.shape[1]))
            token_reps = token_reps[:, token_indices, :]
            token_masks = token_masks[:, token_indices]
           
            '''
            print(span_prompt_len)
            print(prompt_len)
            #to see the actual subword tokens for debugging
            #bring in the raw word_tokens with prompt (called prompt_x in the transform_encoder)
            print(word_tokens[0])    #this one has an early start....
            print([word_tokens[0][i] for i in token_indices if i < len(word_tokens[0])])    #this one has an early start....
            print([word_tokens[0][i] for i in s_token_indices if i < len(word_tokens[0])])
            print([word_tokens[0][i] for i in r_token_indices if i < len(word_tokens[0])])
            '''

        return dict(token_reps      = token_reps,       #(b, max_batch_seq_len (token or subtoken), h)
                    token_masks     = token_masks,      #(b, max_batch_seq_len (token or subtoken))
                    span_type_reps  = span_type_reps,   #(b, num span types, h) => num span types = pos types only for multilabel and pos/neg types for unilabel
                    rel_type_reps   = rel_type_reps,    #(b, num rel types, h) => num rel types = pos types only for multilabel and pos/neg types for unilabel
                    w2sw_map        = w2sw_map_new)     #list of dicts => adjusted to align with token_reps if no pooling used
    


################################################################################################
################################################################################################
################################################################################################
################################################################################################

class RelationProcessor():
    '''
    We run this from the model forward pass as we need the filtered span data as inputs.
    Processes relational data to generate candidate relation tensors. It works with the data
    from `x['relations']` to form tensors based on candidate span IDs.
    '''
    def __init__(self, config):
        '''
        This just accepts the config namespace object
        '''
        self.config = config



    def get_cand_rel_tensors(self, raw_rels, orig_map, cand_span_map, cand_span_labels, cand_span_masks):
        """
        Generates relation labels and masks for all possible pairs of candidate spans derived from provided span indices.
        This function is crucial for dynamic and efficient relation extraction post candidate span filtering, adapting to both unilabel and multilabel scenarios.

        Args:
            raw_rels (list of lists of tuples): Nested list where each inner list corresponds to a batch and contains tuples of the form (head, tail, rel_type_string), representing relationships between spans.
            orig_map (list of dicts): List where each dictionary maps original span indices to new indices within candidate spans for a particular batch.
            cand_span_map (torch.Tensor): Tensor of shape (batch, top_k_spans) that maps candidate span indices to their new tensor indices within the model's processing context.
            cand_span_labels (torch.Tensor): Tensor containing labels for candidate spans, which can be either unilabel (integers) or multilabel (boolean vectors).
            cand_span_masks (torch.Tensor): Boolean tensor indicating valid candidate spans to consider for relation processing.

        Returns:
            Tuple containing three elements:
            - rel_labels (torch.Tensor): Tensor of relation labels. Shape is (batch, top_k_spans**2) for unilabel or (batch, top_k_spans**2, num_rel_types) for multilabel, depending on configuration.
            - rel_masks (torch.Tensor): Boolean tensor of shape (batch, top_k_spans**2) indicating valid relations derived from candidate spans.
            - rel_ids (torch.Tensor): Tensor of shape (batch, top_k_spans**2, 2) storing pairs of indices corresponding to the head and tail of each relation.

        Notes:
            The processing of relations is designed to handle dynamically determined candidate spans, with efficient memory usage in mind. Relations are initialized based on the presence of candidate spans and are processed to exclude self-relations and unsupported mappings. Special attention is required for edge cases where candidate mappings might lead to missing relations, as indicated by the handling of missing indices with a default of -1 (leading to no matches found).

        Example usage:
            This function is typically called after spans are filtered and candidate spans are identified within a model's pipeline. It adjusts relation data to align with these filtered views, ensuring that relation training and inference are based on the currently active span set.
        """
        #get dims of the caididate span tensor
        batch = cand_span_labels.shape[0]
        top_k_spans = cand_span_labels.shape[1]
        device = cand_span_labels.device

        #make the rel masks, ids and labels
        rel_masks = torch.zeros((batch, top_k_spans, top_k_spans), dtype=torch.bool, device=device)
        rel_ids = torch.full((batch, top_k_spans, top_k_spans, 2), 0, dtype=torch.int32, device=device)  # For storing head and tail indices
        if self.config.rel_labels == 'unilabel':
            rel_labels = torch.full((batch, top_k_spans, top_k_spans), 0, dtype=torch.int32, device=device)
        elif self.config.rel_labels == 'multilabel':
            rel_labels = torch.zeros((batch, top_k_spans, top_k_spans, self.config.num_rel_types), dtype=torch.bool, device=device)

        for i in range(batch):
            #go through the relations
            for rel in raw_rels[i]:
                #get the original head and tail span ids and the string rel label
                head_orig, tail_orig, rel_type = rel
                #map these to the cand_span_ids dim 1 with a boolean index first as the head or tail may not be present if we disable pos forcing (i.e. some pos cases are missing from cand_span_ids)
                head_cand_idx = (cand_span_map[i] == orig_map[i].get(head_orig, -1)).nonzero(as_tuple=True)[0]    #cand_span_map is never -1 so if -1 nothing will be found
                tail_cand_idx = (cand_span_map[i] == orig_map[i].get(tail_orig, -1)).nonzero(as_tuple=True)[0]
                # Update rel_labels only if both head and tail indices are part of the candidate spans
                if head_cand_idx.numel() > 0 and tail_cand_idx.numel() > 0:
                    # Update the tensors for labels and IDs
                    rel_masks[i, head_cand_idx, tail_cand_idx] = True
                    rel_ids[i, head_cand_idx, tail_cand_idx, 0] = head_cand_idx.item()
                    rel_ids[i, head_cand_idx, tail_cand_idx, 1] = tail_cand_idx.item()
                    #get the relation label id
                    r_label_id = self.config.r_to_id[rel_type]
                    if self.config.rel_labels == 'unilabel':
                        rel_labels[i, head_cand_idx, tail_cand_idx] = r_label_id
                    elif self.config.rel_labels == 'multilabel':
                        #convert rel label id to a position in the multilabel binary label vector
                        r_label_pos = r_label_id - 1
                        rel_labels[i, head_cand_idx, tail_cand_idx, r_label_pos] = True

        #Mask out self-relations (i.e., diagonal elements)
        diagonal_mask = torch.eye(top_k_spans, device=device, dtype=torch.bool).unsqueeze(0)
        rel_masks = rel_masks & ~diagonal_mask

        #Flatten tensors for compatibility with downstream processing
        rel_masks = rel_masks.view(batch, -1)
        rel_ids = rel_ids.view(batch, -1, 2)
        if self.config.rel_labels == 'unilabel':
            rel_labels = rel_labels.view(batch, -1)
        elif self.config.rel_labels == 'multilabel':
            rel_labels = rel_labels.view(batch, -1, self.config.num_rel_types)

        return rel_labels, rel_masks, rel_ids
    


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







###############################################################
#testing
#YOU NEED TO TEST THIS
###############################################################





class Config:
    def __init__(self):
        # Configuration to map relation types to integers
        self.r_to_id = {'friend': 1, 'enemy': 2, 'colleague': 3}



if __name__ == '__main__':
    # Setup
    config = Config()
    processor = RelationProcessor(config)

    # Test case: No relations
    print("Test 1: No Relations")
    raw_rels = [[]]
    orig_map = [{}]
    cand_span_map = torch.tensor([[0]])
    cand_span_labels = torch.tensor([[1]])
    cand_span_masks = torch.tensor([[True]])
    rel_labels, rel_masks = processor.get_cand_rel_tensors(raw_rels, orig_map, cand_span_map, cand_span_labels, cand_span_masks)
    print("Labels:", rel_labels)
    print("Masks:", rel_masks)

    # Test case: One simple relation
    print("\nTest 2: One Simple Relation")
    raw_rels = [[(0, 2, 'friend'), (1, 0, 'colleague')]]
    orig_map = [{0: 0, 1: 1, 2:2}]
    cand_span_map = torch.tensor([[0, 1, 2]])
    cand_span_labels = torch.tensor([[1, 2, 3]])
    cand_span_masks = torch.tensor([[True, True, True]])
    rel_labels, rel_masks = processor.get_cand_rel_tensors(raw_rels, orig_map, cand_span_map, cand_span_labels, cand_span_masks)
    print("Labels:", rel_labels)
    print("Masks:", rel_masks)

    # Test case: Multiple relations and missing span
    #this shows what happens when some of spans are masked, the mask shoud be replicatedin the rel_mask
    #also shows what happens if a rel pos case is not included in cand_span_map (nothing, it is left out)
    print("\nTest 3: Multiple Relations and Missing Span")
    raw_rels = [[(0, 1, 'friend'), (1, 2, 'enemy'), (0, 3, 'colleague')]]
    orig_map = [{0: 0, 1: 1, 2: 2, 3: 3}]
    cand_span_map = torch.tensor([[0, 1, 2]])
    cand_span_labels = torch.tensor([[1, 2, 3]])
    cand_span_masks = torch.tensor([[True, True, False]])
    rel_labels, rel_masks = processor.get_cand_rel_tensors(raw_rels, orig_map, cand_span_map, cand_span_labels, cand_span_masks)
    print("Labels:", rel_labels)
    print("Masks:", rel_masks)
