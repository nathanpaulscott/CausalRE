from typing import List
import torch
from flair.data import Sentence as flair_sent
from flair.embeddings import TransformerWordEmbeddings as flair_transformer
from .data_processor import PromptProcessor

class TransformerEncoderFlair_w_prompt(torch.nn.Module):
    '''
    This uses the flair implementation of the bert tokenizer and transformer to basically encode the input seq and convert the sw reps to word reps by first sw token selection.  
    In short I do not like it, it is prone to errors and it is inflexible, it can nto support bigbird for example
    Quite simply this whole class can be re-written in python and hugging face to be way better and more flexible!!!!!

    #see docs on the flair bert model implementation
    #https://flairnlp.github.io/docs/tutorial-embeddings/transformer-embeddings
     
    '''
    def __init__(self, config):
        super().__init__()

        self.config = config
        #handle the no subtoken pooling case, reject as flair doesn't support no pooling
        if self.config.subtoken_pooling == 'none':
            raise Exception('flair Error: flair can not do no subtoken pooling, use HF for these cases....')

        #make the prompt processor
        self.prompt_proc = PromptProcessor(config)

        #set the flair params
        flair_params = dict(
            model                = self.config.model_name,
            fine_tune            = self.config.fine_tune,
            subtoken_pooling     = self.config.subtoken_pooling,    #the metod to convert the sw token reps to word token reps, they use first sw token in this case
            layers               = '-1',    #default is 'all' meaning all layer outputs are averaged to get the final transformer output, which is NOT standard, graphER default uses all, which is not good
            layer_mean           = False,          #to not average any layer outputs
            allow_long_sentences = False,   #this should be set to False, True will split sequences longer than 512 sw tokens to get the reps from bert, then the output tensors are concatenated.  The issue is the reps from the split sequences will have no contextual understanding of each other, it is more prudent to set this to False and find a bigger model (bigbird)
        )
        #make the flair transformer encoder
        self.bert_layer = flair_transformer(**flair_params)

        # add tokens to vocabulary
        self.bert_layer.tokenizer.add_tokens([self.config.s_token, self.config.r_token, self.config.sep_token])
        # resize token embeddings
        self.bert_layer.model.resize_token_embeddings(len(self.bert_layer.tokenizer))

        #read the actual model hidden size
        bert_hidden_size = self.bert_layer.embedding_length
        #do the reprojection if required, this would typically only required if subtoken pooling is set to 'first_last'
        if self.config.hidden_size != bert_hidden_size:
            self.projection = torch.nn.Linear(bert_hidden_size, self.config.hidden_size)



    def transformer_encoder(self, tokens: List[List[str]], seq_lengths: torch.Tensor):
        '''
        this runs the word tokens through the bert tokenizer and model and then pools the output sw embeddings back to word embeddings
        thus the output here is a tensor of word embeddings per obs
        NOTE: flair handles tokenizing and putting any relevant tensors on the GPU
        '''
        sentences = [flair_sent(x) for x in tokens]
        self.bert_layer.embed(sentences)
        token_embeddings = torch.nn.utils.rnn.pad_sequence([torch.stack([t.embedding for t in k]) for k in sentences], batch_first=True)
        max_len = seq_lengths.max()
        token_masks = torch.arange(max_len).to(self.device)
        token_masks = (token_masks[None, :] < seq_lengths[:, None]).long()

        #do the reprojection if required, this would typically only required if subtoken pooling was enabled and set to 'first_last'
        if hasattr(self, 'projection'):
            token_embeddings = self.projection(token_embeddings)

        return token_embeddings, token_masks




    def forward(self, x, mode=None):
        '''
        this adds prompt prefix tokens (based on the span types and rel types) to each token seq
        the flair version of bert tokenizes the incoming word tokens to subword tokens 
        => runs through the model => then merges the subword embeddings back to word token embeddings after bert
        '''
        self.device = next(self.parameters()).device
        #add prompt to tokens
        result = self.prompt_proc.add_prompt_to_tokens(x)
        prompt_x        = result['prompt_x']
        len_prompt_x    = result['len_prompt_x']
        prompt_len      = result['prompt_len']
        span_prompt_len = result['span_prompt_len']
        #run through the encoder
        #NOTE: this is currently a flair BERT encoder with additional post processing steps that merge the sw token reps back to word token reps
        #NOTE: it takes in a list of ragged lists of word token strings, converted internally by flair
        embeddings, masks = self.transformer_encoder(prompt_x, len_prompt_x)
        #split the embeddings to: token_reps, token_masks, span_type_reps, rel_type_reps
        #NOTE: token_reps and token_masks will be in word tokens as flair always does subtoken pooling
        #return self.prompt_proc.split_embeddings(embeddings, masks, prompt_len, span_prompt_len)
        result = self.prompt_proc.split_embeddings(embeddings, 
                                                   masks, 
                                                   prompt_len, 
                                                   span_prompt_len, 
                                                   word_tokens = prompt_x)   #for debug only
        #read in values and return
        #NOTE: leave it like this for clarity, doesn't matter if verbose
        token_reps     = result['token_reps'] 
        token_masks    = result['token_masks'] 
        span_type_reps = result['span_type_reps'] 
        rel_type_reps  = result['rel_type_reps']

        #return the w2sw_map as we will need for downstream tasks
        return dict(token_reps     = token_reps, 
                    token_masks    = token_masks, 
                    span_type_reps = span_type_reps, 
                    rel_type_reps  = rel_type_reps,
                    cls_reps       = None,      #will only have value if we are using HF
                    sw_span_ids    = None,      #will only have value if we are in nonpooling mode for HF
                    w2sw_map       = None)      #will only have value if we are in nonpooling mode for HF


