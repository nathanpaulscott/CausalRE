import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from typing import Tuple, List, Dict, Union
from torch.profiler import record_function



class TransformerEncoderHF(torch.nn.Module):
    '''
    This implemements the HF transformer encoder.
    It follows the flair methodology of integrating the tokenizer with the model so that we can dynamically pad in the tokenizer and just make things simpler in general
    This means that the foward method for this class accepts inputs 'x' as a list of ragged lists of word token strings (as opposed to a list of subword tokenized tensors which would be required for pre-model tokenization)
    The performance loss from not splitting the tokenizer is minimal for small to moderate sized datasets.    
    '''
    def __init__(self, config):
        super().__init__()

        self.config = config

        #get the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.backbone_model_name, use_fast=True)
        if not self.tokenizer.is_fast:
            raise Exception("Tokenizer Error: the model you chose doesn't have a fast tokenizer, slow tokenizers are not supported right now for this code....")
        
        #get the model
        self.model = AutoModel.from_pretrained(self.config.backbone_model_name).to(self.config.device)
        if self.config.freeze_encoder:
            for param in self.model.parameters():
                param.requires_grad = False
                
        # Special tokens initialization for markers
        #store the special tokens also
        self.span_start_token = '<span_start>'
        self.span_end_token   = '<span_end>'
        self.head_start_token = '<head_start>'
        self.head_end_token   = '<head_end>'
        self.tail_start_token = '<tail_start>'
        self.tail_end_token   = '<tail_end>'
        special_tokens = [self.span_start_token, self.span_end_token, self.head_start_token, self.head_end_token, self.tail_start_token, self.tail_end_token]
        added_tokens = {'additional_special_tokens': special_tokens}
        self.tokenizer.add_special_tokens(added_tokens)
        self.model.resize_token_embeddings(len(self.tokenizer))

        #Store special token IDs
        tokens_ids = self.tokenizer.convert_tokens_to_ids(special_tokens)
        (self.span_start_id, self.span_end_id,
         self.head_start_id, self.head_end_id,
         self.tail_start_id, self.tail_end_id) = tokens_ids
        self.cls_token_id = self.tokenizer.cls_token_id
        self.sep_token_id = self.tokenizer.sep_token_id

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



    def get_w2sw_map_fast_tokenizer(self, encodings):
        '''
        Determine the w2sw_map (word token to subword token mapping dict)
        NOTE: w2sw_map will only have keys as word tokens (no special tokens as they are only sw tokens)
        for the fast tokenizer case which uses word_ids.
        Operates on a batch of data, w2sw_map will have one map per obs in the batch.
        '''
        batch_size = encodings.input_ids.shape[0]
        w2sw_map = []
        #Create the list of dictionaries version
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


    def make_sw_span_ids(self, span_ids, w2sw_map, max_seq_len):
        '''
        Makes the sw_span_ids which has the sw token span ids for each span, has 0 if the sw token idx is invalid (for a masked out span)

        Args:
            span_ids (torch.Tensor): Tensor of shape (batch_size, num_spans, 2) containing word-level start and end indices
                                    for each span.
            w2sw_map (list): A list of dictionaries, one for each batch item, mapping word token indices to lists of corresponding subword indices.
            max_seq_len (int): the max_seq_length of the sequences in the batch, so basically the dim 1 of token reps tensor

        Returns:
            torch.Tensor: A tensor of the same shape as span_ids, containing mapped subword token indices, with shape
                        (batch_size, num_spans, 2). Each span is represented by start and end indices in subword tokens.
                        Invalid spans (e.g., where the end of the span falls outside the sequence length) have an end subword index
                        of 0. These spans should be masked out in subsequent processing steps.
        '''
        if self.config.subtoken_pooling != 'none':
            return None

        #make the w2sw_tensor with extended seq dim to account for span end indices that go beyond the max_seq_len by max_span_width   
        #these are lookup w index to sw index so the end is actual not end+1
        batch_size = len(w2sw_map)
        extended_seq_dim = max_seq_len + self.config.max_span_width
        #init the tensor with -1s
        w2sw_tensor = torch.full((batch_size, extended_seq_dim, 2), -1, device=self.config.device, dtype=torch.long)
        #fill the tensor
        for i, map in enumerate(w2sw_map):
            for word_id, sw_ids in map.items():
                w2sw_tensor[i, word_id, 0] = min(sw_ids)  # First subword index
                w2sw_tensor[i, word_id, 1] = max(sw_ids)  # Last subword index

        #make the span start/end indices (sw token aligned)
        #NOTE: the lookup w token indices need to be actual (so subtract 1), then we must add 1 to the output sw token end indices 
        #as it will be used to make the final output, which must be in (start, end + 1) format
        sw_start_indices = torch.gather(w2sw_tensor[:, :, 0], 1, span_ids[:, :, 0])
        #NOTE: for end indices, clamp to 0 to limit the invalid end indices in span_ids (usually 0) from going to -1, which will cause errors
        sw_end_indices = torch.gather(w2sw_tensor[:, :, 1], 1, torch.clamp(span_ids[:, :, 1] - 1, min=0)) + 1
        #Put together the new sw_span_ids tensor
        sw_span_ids = torch.stack([sw_start_indices, sw_end_indices], dim=-1)

        return sw_span_ids     #(batch, max_num_spans_batch, 2)
    

    def subtoken_pooler(self, embeddings, masks, w2sw_map):
        """
        NOTE: [CLS] and [SEP] reps have been removed from embedings, masks and w2sw_map before this function
        embeddings: tensor of subtoken embeddings of shape (batch, max_batch_subtoken_seq_len, hidden)
        masks: tensor of shape (batch, max_batch_subtoken_seq_len)
        NOTE: max_seq_len is the max subtoken seq len for the batch
        w2sw_map: list of dicts, one for each obs, mapping word_idx -> list of subword_indices
        
        outputs: 
        - token_embeddings: word_tokens embeddings
        - token_masks: attention mask for word tokens (as dtype bool)
        NOTE: for the first_last case the hidden dim will be doubled, which is handled by the projection code to pull it back to normal
        NOTE: the outputs are padded to the max word token seq length for the batch
        """
        batch_size = embeddings.size(0)
        max_word_tokens = max(max(d.keys()) for d in w2sw_map) + 1  # +1 adjusting 0-based index
        hidden_size = embeddings.size(-1)
        
        output_size = hidden_size * 2 if self.config.subtoken_pooling == 'first_last' else hidden_size
        #these are the word token embeddings and masks tensors that we will put the pooled data in, they will not include the cls/sep tokens
        token_embeddings = torch.zeros(batch_size, max_word_tokens, output_size, device=self.config.device)
        token_masks = torch.zeros(batch_size, max_word_tokens, dtype=torch.bool, device=self.config.device)

        for b in range(batch_size):
            # Pool the regular word tokens
            #NOTE: subtoken indices have been adjusted to account for no [CLS] token
            for word_idx, sw_idxs in w2sw_map[b].items():
                # Skip if any subtoken is invalid
                if not all(masks[b, idx] for idx in sw_idxs):
                    continue

                #is good so set the word mask to True
                token_masks[b, word_idx] = True
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
        


    def set_window_attention_masks_rel(self, input_ids, attention_masks, window):
        batch_size, seq_len = input_ids.shape
        #Start with the original attention mask
        modified_attention_masks = attention_masks.clone()
        #Identify CLS and SEP token pos (always allow full attention)
        cls_mask = (input_ids == self.cls_token_id)
        sep_mask = (input_ids == self.sep_token_id)
        special_tokens_mask = cls_mask | sep_mask
        modified_attention_masks[special_tokens_mask] = 1  # Always unmask CLS & SEP tokens

        #Process each batch separately
        for b in range(batch_size):
            # Get indices for relation head/tail markers
            head_start_pos = (input_ids[b] == self.head_start_id).nonzero(as_tuple=True)[0]
            head_end_pos = (input_ids[b] == self.head_end_id).nonzero(as_tuple=True)[0]
            tail_start_pos = (input_ids[b] == self.tail_start_id).nonzero(as_tuple=True)[0]
            tail_end_pos = (input_ids[b] == self.tail_end_id).nonzero(as_tuple=True)[0]
            # No relations to process in this batch
            if len(head_start_pos) == 0 or len(tail_start_pos) == 0:
                continue
            allowed_mask_obs = torch.zeros_like(attention_masks[b])
            # Unmask CLS and SEP
            allowed_mask_obs[special_tokens_mask[b]] = 1
            # Process head and tail markers
            for head_start, head_end, tail_start, tail_end in zip(head_start_pos, head_end_pos, tail_start_pos, tail_end_pos):
                # Ensure each segment between starts and ends is unmasked
                allowed_mask_obs[head_start:head_end + 1] = 1
                allowed_mask_obs[tail_start:tail_end + 1] = 1
                # Apply window around the head and tail segments
                head_window_start = torch.clamp(head_start - window, min=0)
                head_window_end = torch.clamp(head_end + window + 1, max=seq_len)
                tail_window_start = torch.clamp(tail_start - window, min=0)
                tail_window_end = torch.clamp(tail_end + window + 1, max=seq_len)
                allowed_mask_obs[head_window_start:head_start + 1] = 1  # Window before head start
                allowed_mask_obs[head_end:head_window_end] = 1  # Window after head end
                allowed_mask_obs[tail_window_start:tail_start + 1] = 1  # Window before tail start
                allowed_mask_obs[tail_end:tail_window_end] = 1  # Window after tail end

            # Apply this batch's mask to the original
            modified_attention_masks[b] = modified_attention_masks[b] & allowed_mask_obs

        return modified_attention_masks



    def set_window_attention_masks_span(self, input_ids, attention_masks, window):
        """
        Modify the existing 1D attention mask to apply a local attention window around marked spans.

        Args:
            input_ids (torch.Tensor): (batch, seq_len) Tokenized input tensor.
            attention_masks (torch.Tensor): (batch, seq_len) Original attention mask.
            window (int): Size of the attention window.

        Returns:
            torch.Tensor: (batch, seq_len) modified attention mask.
        """
        batch, seq_len = input_ids.shape
        #Start with the original attention mask
        modified_attention_masks = attention_masks.clone()
        #Identify CLS and SEP token pos (always allow full attention)
        cls_mask = (input_ids == self.cls_token_id)
        sep_mask = (input_ids == self.sep_token_id)
        special_tokens_mask = cls_mask | sep_mask
        modified_attention_masks[special_tokens_mask] = 1  # Keep CLS & SEP unmasked
        #Process each batch separately
        for b in range(batch):
            #Get indices for span start and end
            start_pos = (input_ids[b] == self.span_start_id).nonzero(as_tuple=True)[0]
            end_pos = (input_ids[b] == self.span_end_id).nonzero(as_tuple=True)[0]
            #Start with a blank mask (disable attention everywhere except CLS/SEP)
            allowed_mask_obs = torch.zeros_like(attention_masks[b])
            allowed_mask_obs[special_tokens_mask[b]] = 1  # Keep CLS/SEP unmasked
            #Enable local attention within spans
            for start, end in zip(start_pos, end_pos):
                #Enable full self-attention inside the span
                allowed_mask_obs[start:end+1] = 1  
                #Enable the window region before and after the span
                left_bound = torch.clamp(start - window, min=0)
                right_bound = torch.clamp(end + window + 1, max=seq_len)
                allowed_mask_obs[left_bound:start+1] = 1  # Backward window
                allowed_mask_obs[end:right_bound] = 1  # Forward window
            #Apply this mask to the batch
            modified_attention_masks[b] = modified_attention_masks[b] & allowed_mask_obs

        return modified_attention_masks


    def transformer_encoder_basic(self, tokens, type, window=None):
        '''
        Runs both the tokenizer and encoder part of the HF transformer encoder.
        Implements **windowed attention**, limiting token attention to within `window_size`.

        Args:
            tokens (List[List[str]]): Batch of tokenized sequences.
            window_size (int, optional): The max distance for local attention. If None, default full attention.  units are sw tokens

        Returns:
            dict: Containing input_ids and embeddings.
        '''
        #run the tokenizer
        ##############################
        result = self.tokenizer(
            text =                  tokens,
            is_split_into_words =   True,   #indicates that text is already word tokenized
            padding =               True,   #will pad to the longest seq in the batch => dynamic padding as we are operating per batch here, this is one advantage of integrating the tokenizer with the model
            truncation =            True,   #will truncate to the max allowable subtoken len for the model => no need to specify 'max_length'
            add_special_tokens=     True,   #not needed as default is True
            return_offsets_mapping= False,
            return_tensors =        'pt'
        )
        #get the input_ids and attention mask
        input_ids = result['input_ids'].to(self.config.device)     #these have the CLS and SEP tokens in them
        attention_masks = result['attention_mask'].to(self.config.device)    #these have the CLS and SEP tokens in them, comes out of the tokenizer as torch.int64/long
        ##############################

        #modify the attention masks to ignore tokens beyond a window from the marked span
        if window is not None and window != 'none':
            if type == 'span':
                attention_masks = self.set_window_attention_masks_span(input_ids, attention_masks, window)
            else:
                attention_masks = self.set_window_attention_masks_rel(input_ids, attention_masks, window)
                
        #run the transformer encoder model to generate the subtoken embeddings
        ##############################
        result = self.model(input_ids=input_ids, attention_mask=attention_masks)
        embeddings = result.last_hidden_state
        ##############################

        return dict(input_ids  = input_ids, 
                    embeddings = embeddings)



    def transformer_encoder(self, tokens):
        '''
        runs both the tokenizer and encoder part of the HF transformer encoder        
        This function operates on a batch
        '''
        #run the tokenizer
        ##############################
        try:
            result = self.tokenizer(
                text =                  tokens,
                is_split_into_words =   True,   #indicates that text is already word tokenized
                padding =               True,   #will pad to the longest seq in the batch => dynamic padding as we are operating per batch here, this is one advantage of integrating the tokenizer with the model
                truncation =            True,   #will truncate to the max allowable subtoken len for the model => no need to specify 'max_length'
                add_special_tokens=     True,   #not needed as default is True
                return_offsets_mapping= True,
                return_tensors =        'pt'
            )
        except Exception as e:
            print('\nThese tokens caused issues with your tokenizer, check the inputs...')
            print(tokens)
            print(e)
            print('Exiting.....')
            exit()
        #get the input_ids and attention mask
        input_ids = result['input_ids'].to(self.config.device)     #these have the CLS and SEP tokens in them
        masks = result['attention_mask'].to(self.config.device)    #these have the CLS and SEP tokens in them, comes out of the tokenizer as torch.int64/long
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
        masks = masks[:, 1:-1].to(dtype=torch.bool)    #change masks to bool
        #remove 1 from the sw token idx in w2sw_map to account for removing [CLS], [SEP] tokens
        w2sw_map = [{k:[i-1 for i in v] for k,v in x.items()} for x in w2sw_map]
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

        return dict(embeddings = embeddings, 
                    masks      = masks, 
                    cls_reps   = cls_reps,
                    w2sw_map   = w2sw_map)



    def forward(self, x):
        '''
        this runs the bert tokenizer and transformer
        it then merges the subword embeddings back to word token embeddings after bert if required
        '''
        ################################################
        span_ids        = x['span_ids']
        tokens          = x['tokens']
        #token_lengths   = x['seq_length']
        #max_seq_len     = x['seq_length'].max().item()
        ################################################
        
        ###################################################
        #Run the modified or unmodified inputs through the HF encoder transformer
        result = self.transformer_encoder(tokens)
        #read in the results
        embeddings  = result['embeddings']
        masks       = result['masks']
        cls_reps    = result['cls_reps']
        #w2sw_map    = result['w2sw_map']
        ###################################################

        ################################################
        #Init the outputs
        token_reps     = embeddings
        token_masks    = masks
        ################################################
 
        ################################################
        #make the sw_span_ids tensor for the batch (returns None if we have pooling)
        #sw_span_ids = self.make_sw_span_ids(span_ids, w2sw_map, max_seq_len)
        ################################################
 
        #Return the results based on whether prompts were used
        return dict(
            token_reps      = token_reps,
            token_masks     = token_masks,
            cls_reps        = cls_reps,
            #sw_span_ids     = sw_span_ids,     # Will only have value if in non-pooling mode for HF
        )