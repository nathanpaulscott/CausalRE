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


    def batch_list_to_dict_converter(self, batch: List[Dict]) -> Dict:
        '''
        converts a batch from a list of dicts (one dict per obs) to a dict of various things
        i.e. The values for each key could be lists, tensors, or dicts depending on the usage, 
        eg. 
        s_to_id is a dict, 
        tokens is a list of ragged lists, 
        span_ids is a tensor
        spans is a list of tuples

        NOTE:
        All these properties are set in the dataprocessor __init__ method
        self.config.s_to_id, self.config.id_to_s, self.config.r_to_id, self.config.id_to_r
        '''
        #convert seq_length to a tensor
        seq_length = torch.tensor([x["seq_length"] for x in batch], dtype=torch.long)
        #pad the span_idx and span_label list of tensors to the longest observation in the batch and convert the whole thing to a tensor
        #READ THIS
        #remember: obs['span_ids'] and obs['span_labels'] are aligned to all possible spans (self.config.all_span_ids)
        #obs['span_ids'] are just ints, so the padding here is to fill up where the seq_len if different between obs
        #obs['span_labels'] are ints from 0 to num_span_types, 0 means non-span, > 0 means a classified span and -1 is an invalid span, thus we further pad with more -1 to deal with the different seq_lens.  
        #the key point is that span_labels == 0 are neg samples, span_labels > 0 are pos samples, span_labels == -1 are invalid spans or padding => invlaid spans were marked -1 in labels back in the preprocessing function
        #so at this point the span_mask info is in the span_labels as -1
        #READ THIS
        span_ids = torch.nn.utils.rnn.pad_sequence([obs["span_ids"] for obs in batch], batch_first=True, padding_value=0)
        span_label = torch.nn.utils.rnn.pad_sequence([obs["span_label"] for obs in batch], batch_first=True, padding_value=-1)
        #we pull out the span_mask info from the span_labels into it's own tensor for clarity, mask = 1 for good spans (pos and neg cases) and 0 for ignore spans (padding or invalid)
        span_mask = span_label != -1
        #these ones are just lists
        tokens = [obs["tokens"] for obs in batch]
        spans = [obs["spans"] for obs in batch]
        relations = [obs["relations"] for obs in batch]

        # Return a dict of lists
        return dict(
            tokens =        tokens,         #list of ragged lists of strings => the raw word tokenized seq data
            spans =         spans,          #list of ragged list of tuples => the positive cases for each obs 
            relations =     relations,      #list of ragged list of tuples => the positive cases for each obs
            seq_length =    seq_length,     #tensor (batch) the length of tokens for each obs
            span_ids =      span_ids,       #tensor (batch, max_seq_len_batch*max_span_width, 2) => the span_ids truncated to the max_seq_len in the batch* max_span_wdith
            span_mask =     span_mask,      #tensor (batch, max_seq_len_batch*max_span_width) => 1 for valid spans, 0 for pad and invalid spans
            span_label =    span_label,     #tensor (batch, max_seq_len_batch*max_span_width) => 0 to num_span_types for valid cases, -1 for invalid and pad cases
        )



    #remember staticmethod is a python decorator that just ties the function to the class not the instance
    #as a result it doesn't need the self part, but this also means it doesn't have access to self
    @staticmethod
    def make_span_to_label_defaultdict(
        spans: List[Tuple[int, int, str]], 
        span_types_to_id: Dict[str, int]) -> Dict[Tuple[int, int], int]:
        '''
        This is just a utility function, probably should be utilities, it is for looking up the span_type idx from the span (start, end) tuple, it returns 0 if the given tuple is not there
        NOTE: We already have s_to_id which is span_type string to span_type idx, but this one is for span (start, end) tuple to span idx

        Makes a defaultdict to map the span (start, end) tuple to the label idx.
        NOTE: the default dict functionality is crucial as it returns 0 for keys that are NOT found in the dict
        0 is the label index for the none_span, i.e. span that doesn't represent an entity/event etc..

        Input spans is a list of tuples [(start, end, label)...]
        Output is a defaultdict mapping (start, end) to label_idx, where label_idx is 1 onwards, returns 0 if key not in the dict
        '''
        span_map = defaultdict(int)
        for span in spans:
            if span[-1] not in span_types_to_id or span_types_to_id[span[-1]] == 0:    #the span_type was not found or it maps to the none_span, do not include as we will rely on the defaultdict behaviour to return idx 0
                continue
            span_map[(span[0], span[1])] = span_types_to_id[span[-1]]
        return span_map



    def preprocess_obs(self, obs: Dict) -> Dict:
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

        #simplify the spans and rels to list of tuples
        spans = [(x['start'], x['end'], x['type']) for x in obs['spans']]
        rels = [(x['head'], x['tail'], x['type']) for x in obs['relations']]

        #get all possible spans in seq_len, noting there will be some who's 'end' goes past the seq_len bounds
        #length of span_ids will be seq_len*max_span_width
        span_ids = [x for x in self.config.all_span_ids if x[0] < seq_len]
        # Get the dictionary of span labels key is (start, end), value is the class as idx
        #this maps the (start, end) to the label idx, returns 0 if key not in dict (defaultdict behaviour)
        label_dict = self.make_span_to_label_defaultdict(spans, self.config.s_to_id) if spans else defaultdict(int)
        span_labels = [label_dict[x] for x in span_ids]   #label of 0 if the span doesn't have a label as label_dict is a default dict (returns 0 to keys that are not there)
        #move to tensors
        span_labels = torch.tensor(span_labels, dtype=torch.long)
        span_ids = torch.tensor(span_ids, dtype=torch.long)
        #mask out the invalid spans from the span_labels by setting the label to -1
        #this is the masking of those spans who end outside the seq_len, i.e. the invalid spans are set to mask = True
        invalid_span_mask = span_ids[:, 1] > seq_len     #no -1 here as we have pythonic end indices
        span_labels = span_labels.masked_fill(invalid_span_mask, -1)

        # Return a dictionary with the preprocessed observations
        return dict(
            tokens =        tokens,             #the word tokens of the input seq
            spans =         spans,              #the simplified list of span tuples [(start, end, type), ...]
            relations =     rels,               #the simplified list of rel tuples [(head, tail, type), ...]
            span_ids =      span_ids,           #tensor (seq_len*max_span_width, 2) all possible span (start,end) tuples starting within tokens
            span_label =    span_labels,        #tensor (seq_len*max_span_width) the span labels aligning with each element in span_idx, 0 if none_span and -1 if the span is invalid
            seq_length =    seq_len,            #length of tokens, a scalar
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


    def create_dataloaders(self, data: dict, **kwargs) -> Dict:
        """
        Create DataLoaders for the dataset with span and relation types extracted from the schema.
        Args:
            data: The dataset to be loaded with train, val, test keys
            **kwargs: Additional arguments passed to the DataLoader.
        Returns:
            one DataLoader per data key: A PyTorch DataLoader instance.
        """
        #make the loaders    
        loaders = dict(
            train = DataLoader(data['train'], collate_fn=self.collate_fn, batch_size=self.config.train_batch_size, shuffle=self.config.shuffle_train, **kwargs),
            val =   DataLoader(data['val'],   collate_fn=self.collate_fn, batch_size=self.config.eval_batch_size,  shuffle=False, **kwargs),
            test =  DataLoader(data['test'],  collate_fn=self.collate_fn, batch_size=self.config.eval_batch_size,  shuffle=False, **kwargs)
        )
        return loaders










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
        This adds in prompt tokens for each span and rel type to the tokens for each obs
        x is the batch, x['tokens'] is a list of lists, the outer list is for the observations, the inner list is for tokens in the sequence
        eg.
        tokens_w_prompt = ['<<S>>', 'span_type1', '<<S>>', 'span_type2', '<<SEP>>', '<<R>>','rel_type1','<<R>>','rel_type2', <<SEP>>, normal input tokens...]
        
        NOTE: self.config.span_types and self.config.rel_types do not include the none_span and none_relation
        
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
                    span_type_reps  = span_type_reps,   #(b, num span types, h)
                    rel_type_reps   = rel_type_reps,    #(b, num rel types, h)
                    w2sw_map        = w2sw_map_new)     #list of dicts => adjusted to align with token_reps if no pooling used