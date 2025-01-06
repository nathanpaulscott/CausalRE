from typing import Tuple, List, Dict, Union
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from .data_processor import PromptProcessor


class TransformerEncoderHFPrompt(torch.nn.Module):
    '''
    This implemements the HF transformer encoder with prompt addition and removal.
    It follows the flair methodology of integrating the tokenizer with the model so that we can dynamically pad in the tokenizer and just make things simpler in general
    This means that the foward method for this class accepts inputs 'x' as a list of ragged lists of word token strings (as opposed to a list of subword tokenized tensors which would be required for pre-model tokenization)
    The performance loss from not splitting the tokenizer is minimal for small to moderate sized datasets.    
    '''
    def __init__(self, config):
        super(TransformerEncoderHFPrompt).__init__()

        self.config = config

        #get the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name, use_fast=True)
        if not self.tokenizer.is_fast:
            raise Exception("Tokenizer Error: the model you chose doesn't have a fast tokenizer, slow tokenizers are not supported right now for this code....")
        
        #get the model
        self.model = AutoModel.from_pretrained(self.config.model_name)
        if not self.config.freeze_encoder:
            for param in self.model.parameters():
                param.requires_grad = False
                
        #add the special tokens to the tokenizer and model if we are using a prompt
        if self.config.use_prompt:
            #add special tokens to the tokenizer and model
            self.tokenizer.add_special_tokens({'additional_special_tokens': [self.config.s_token, self.config.r_token, self.config.sep_token]})
            self.model.resize_token_embeddings(len(self.tokenizer))
            self.config.s_token_id = self.tokenizer.convert_tokens_to_ids(self.config.s_token)
            self.config.r_token_id = self.tokenizer.convert_tokens_to_ids(self.config.r_token)
            #make the prompt processor
            self.prompt_proc = PromptProcessor(config)

        #determine whether we need to add a projection layer after the model
        #NOTE: this is different to the flair model due to how we do the subtoken pooling here
        #this would typically only required if subtoken pooling was enabled and set to 'first_last'
        #and/or the user set the hidden_size parameter something other than the standard transformer encoder value
        bert_hidden_size = self.model.config.hidden_size
        if self.config.subtoken_pooling == 'first_last':
            bert_hidden_size = bert_hidden_size*2
        #do the reprojection if required
        if self.config.hidden_size != bert_hidden_size:
            self.projection = torch.nn.Linear(bert_hidden_size, self.config.hidden_size)



    def get_w2sw_map_fast_tokenizer(self, encodings) -> List[Dict[int, List[int]]]:
        '''
        determine the w2sw_map (word token to subword token mapping dict)
        for the fast tokenizer case which uses word_ids
        operates on a batch of data, w2sw_map will have one map per obs in the batch
        '''
        batch_size = encodings.input_ids.shape[0]
        w2sw_map = []
        for batch_idx in range(batch_size):
            word_ids = encodings.word_ids(batch_idx)  # None for special tokens
            curr_map = {}
            for sw_idx, word_idx in enumerate(word_ids):
                if word_idx is not None:  # Skip special tokens
                    if word_idx not in curr_map:
                        curr_map[word_idx] = []
                    curr_map[word_idx].append(sw_idx)
            w2sw_map.append(curr_map)
        
        return w2sw_map



    def subtoken_pooler(self, embeddings, masks, w2sw_map):
        """
        NOTE: [CLS] and [SEP] reps have been removed from embeedings, masks and w2sw_map before this function
        embeddings: tensor of subtoken embeddings of shape (batch, max_batch_subtoken_seq_len, hidden)
        masks: tensor of shape (batch, max_batch_subtoken_seq_len)
        NOTE: max_seq_len is the max subtoken seq len for the batch
        w2sw_map: list of dicts, one for each obs, mapping word_idx -> list of subword_indices
        
        outputs: 
        - token_embeddings: word_tokens embeddings
        - token_masks: attention mask for word tokens
        NOTE: for the first_last case the hidden dim will be doubled, which is handled by the projection code to pull it back to normal
        NOTE: the outputs are padded to the max word token seq length for the batch
        """
        batch_size = embeddings.size(0)
        max_word_tokens = max(max(d.keys()) for d in w2sw_map) + 1  # +1 adjusting 0-based index
        hidden_size = embeddings.size(-1)
        
        output_size = hidden_size * 2 if self.config.subtoken_pooling == 'first_last' else hidden_size
        #these are the word token embeddings and masks tensors that we will pu tht epooled data in, they will not include the cls/sep tokens
        token_embeddings = torch.zeros(batch_size, max_word_tokens, output_size, device=embeddings.device)
        token_masks = torch.zeros(batch_size, max_word_tokens, dtype=torch.long, device=embeddings.device)

        for b in range(batch_size):
            # Pool the regular word tokens
            #NOTE: subtoken indices have been adjusted to account for no [CLS] token
            for word_idx, sw_idxs in w2sw_map[b].items():
                # Skip if any subtoken is invalid
                if not all(masks[b, idx] == 1 for idx in sw_idxs):
                    continue

                #is good so set the word mask to 1
                token_masks[b, word_idx] = 1
                #now pool the sw tokens
                if self.config.subtoken_pooling == 'first':
                    token_embeddings[b, word_idx] = embeddings[b, sw_idxs[0]] 

                elif self.config.subtoken_pooling == 'last':
                    token_embeddings[b, word_idx] = embeddings[b, sw_idxs[-1]] 

                elif self.config.subtoken_pooling == 'first_last':
                    token_embeddings[b, word_idx] = torch.cat([
                        embeddings[b, sw_idxs[0]],         
                        embeddings[b, sw_idxs[-1]]         
                    ])

                elif self.config.subtoken_pooling == 'mean':
                    token_embeddings[b, word_idx] = embeddings[b, sw_idxs].mean(0)

                elif self.config.subtoken_pooling == 'maxpool':
                    token_embeddings[b, word_idx] = embeddings[b, sw_idxs].max(0)[0]
            
        return token_embeddings, token_masks
        

    def make_sw_span_ids(self, span_ids, w2sw_map):
        '''
        Maps word token start/end spans to subword token start/end spans.
        
        Args:
            span_ids (torch.Tensor): Shape (batch_size, max_batch_seq_len*max_span_width, 2) containing
                                    word-level start and end indices.
            w2sw_map (List[Dict[int, List[int]]]): List of dicts mapping word idx -> subword indices,
                                                one dict per batch.
        Returns:
            torch.Tensor: Same shape as span_ids with mapped subword indices.
        '''
        if self.config.subtoken_pooling != 'none':
            return None

        batch_size, num_spans, _ = span_ids.shape
        
        #init the sw_span_ids
        sw_span_ids = torch.zeros_like(span_ids) 
        for b in range(batch_size):
            for s in range(num_spans):
                start, end = span_ids[b, s, 0], span_ids[b, s, 1]
                # Skip padding and problematic spans
                if (start == 0 and end == 0) or start not in w2sw_map[b] or end-1 not in w2sw_map[b]:
                    continue
                # Map start to first subword token
                sw_span_ids[b, s, 0] = w2sw_map[b][start][0]
                # Map end-1 to last subword token (+1 for exclusive end)
                sw_span_ids[b, s, 1] = w2sw_map[b][end-1][-1] + 1
        
        return sw_span_ids




    def transformer_encoder(self, tokens: List[List[str]], seq_lengths: torch.Tensor):
        '''
        runs both the tokenizer and encoder part of the HF transformer encoder        
        This function operates on a batch
        '''
        #run the tokenizer
        ##############################
        result = self.tokenizer(
            text =                  tokens,
            is_split_into_words =   True,   #indicates that text is already word tokenized
            padding =               True,   #will pad to the longest seq in the batch => dynamic padding as we are operating per batch here, this is one advantage of integrating the tokenizer with the model
            truncation =            True,   #will truncate to the max allowable subtoken len for the model => no need to specify 'max_length'
            add_special_tokens=     True,   #not needed as default is True
            return_offsets_mapping= True,
            return_tensors =        'pt'
        )
        #get the input_ids and attention mask
        input_ids = result['input_ids'].to(self.device)     #this have the CLS and SEP tokens in them
        masks = result['attention_mask'].to(self.device)    #this have the CLS and SEP tokens in them
        #get the word token to subword token map (w2sw_map) using HuggingFace fast tokenizer word_ids()
        w2sw_map = self.get_w2sw_map_fast_tokenizer(result)
        ##############################

        #run the transformer encoder model to generate the subtoken embeddings
        ##############################
        result = self.model(input_ids=input_ids, attention_mask=masks)
        embeddings = result.last_hidden_state
        ##############################

        #extract the [CLS] reps
        ##############################
        cls_reps = embeddings[:, 0, :]
        #remove [CLS] and[SEP] reps from the embeddings and masks, inputs, w2sw_map etc
        embeddings = embeddings[:, 1:-1, :]
        masks = masks[:, 1:-1]
        #remove 1 from the sw token idx in w2sw_map to account for removing [CLS], [SEP] tokens
        w2sw_map = [{k:[i-1 for i in v] for k,v in x.items()} for x in w2sw_map]
        #remove [CLS],[SEP] from input_ids
        input_ids = input_ids[:, 1:-1]
        ##############################

        #do the subtoken pooling if required, i.e. if we want the encoder output to be mapped back to word tokens
        ##############################
        if self.config.subtoken_pooling != 'none':
            embeddings, masks = self.subtoken_pooler(embeddings, masks, w2sw_map)
            w2sw_map = None     #do not need w2sw_map anymore
        ##############################

        #do the reprojection if required, this would typically only required if subtoken pooling was enabled and set to 'first_last'
        ##############################
        if hasattr(self, 'projection'):
            embeddings = self.projection(embeddings)
        ##############################
        
        return dict(input_ids   = input_ids, 
                    embeddings  = embeddings, 
                    masks       = masks, 
                    cls_reps    = cls_reps,
                    w2sw_map    = w2sw_map)


    def forward(self, x):
        '''
        this adds prompt prefix tokens (based on the span types and rel types) to each token seq
        the flair version of bert tokenizes the incoming word tokens to subword tokens 
        => runs through the model => then merges the subword embeddings back to word token embeddings after bert
        '''
        self.device = next(self.parameters()).device
        
        ################################################
        span_ids        = x['span_ids']
        tokens          = x['tokens']
        token_lengths   = x['seq_length']
        prompt_len      = 0
        span_prompt_len = 0
        if self.config.use_prompt:
            # Add prompt to the inputs if use_prompt is True
            result = self.prompt_proc.add_prompt_to_tokens(x)
            #read in the results
            tokens          = result['prompt_x']
            token_lengths   = result['len_prompt_x']
            prompt_len      = result['prompt_len']
            span_prompt_len = result['span_prompt_len']
        ################################################

        ###################################################
        #Run the modified or unmodified inputs through the HF encoder transformer
        result = self.transformer_encoder(tokens, token_lengths)
        #read in the results
        input_ids   = result['input_ids']
        embeddings  = result['embeddings']
        masks       = result['masks']
        cls_reps    = result['cls_reps']
        w2sw_map    = result['w2sw_map']
        ###################################################

        ################################################
        #Init the outputs for the no prompt case
        token_reps     = embeddings
        token_masks    = masks
        span_type_reps = None
        rel_type_reps  = None
        if self.config.use_prompt:
            # Split the embeddings into different representations if prompts were used
            result = self.prompt_proc.split_embeddings(embeddings, 
                                                       masks, 
                                                       prompt_len, 
                                                       span_prompt_len,
                                                       w2sw_map, 
                                                       input_ids, 
                                                       tokens, 
                                                       self.tokenizer)
            #read in results
            token_reps     = result['token_reps']
            token_masks    = result['token_masks']
            span_type_reps = result['span_type_reps']
            rel_type_reps  = result['rel_type_reps']
            w2sw_map       = result['w2sw_map']
        ################################################
 
        ################################################
        #make the sw_span_ids tensor for the batch (returns None if we have pooling)
        sw_span_ids = self.make_sw_span_ids(span_ids, w2sw_map)
        ################################################
 
        #Return the results based on whether prompts were used
        return dict(
            token_reps      = token_reps,
            token_masks     = token_masks,
            span_type_reps  = span_type_reps,
            rel_type_reps   = rel_type_reps,
            cls_reps        = cls_reps,
            sw_span_ids     = sw_span_ids,     # Will only have value if in non-pooling mode for HF
            w2sw_map        = w2sw_map         # Will only have value if in non-pooling mode for HF
        )