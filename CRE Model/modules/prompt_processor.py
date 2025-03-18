from collections import defaultdict
from typing import Tuple, List, Dict, Union

import torch
from torch.utils.data import DataLoader
import random



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
            masks: Tensor of shape (batch, seq_len) containing bool attention masks with [CLS], [SEP] removed
            prompt_len: Integer length of full prompt in word tokens
            span_prompt_len: Integer length of span prompt portion in word tokens 
            w2sw_map: Optional list of dicts mapping word indices to subword indices, adjusted for [CLS], [SEP] removal
            input_ids: Optional tensor of input token ids with [CLS], [SEP] removed
            word_tokens: Optional list of original word tokens including prompt
            tokenizer: Optional tokenizer object for debugging

        Returns:
            Dictionary containing:
            - token_reps: Tensor (batch, max_seq_len, hidden) of non-prompt token representations
            - token_masks: Tensor (batch, max_seq_len) of attention masks for non-prompt tokens, dttype bool
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

        else:     #pooling case, working in word tokens
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
    

