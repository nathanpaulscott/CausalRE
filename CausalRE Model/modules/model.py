import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as init
from torch.nn import CrossEntropyLoss
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch.autograd import profiler

from pathlib import Path

###############################################
#custom imports
from .layers_transformer_encoder_flair import TransformerEncoderFlairPrompt
from .layers_transformer_encoder_hf import TransformerEncoderHFPrompt
from .layers_other import FFNProjectionLayer, LstmSeq2SeqEncoder, TransformerEncoderTorch, GraphEmbedder
from .loss_functions import compute_matching_loss
from .span_rep import SpanRepLayer
from .filtering import FilteringLayer
from .data_processor import RelationProcessor
from .rel_rep import RelationRepLayer
from .scorer import ScorerLayer
from .utils import er_decoder, get_relation_with_span, load_from_json, save_to_json
from .evaluator import Evaluator



'''
NATHAN
--------------------
So this creates the model class and imports almost all modules in the modules folder

This module is imported by train.py and the model object is instantiated from there

NOTE: I took out the base part that was in graphER and pu thtat functionality in here, except the HF mixin stuff
'''


class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        #special token definition
        self.config.s_token = "<<S>>"
        self.config.r_token = "<<R>>"
        self.config.sep_token = "<<SEP>>"

        ######################################
        
        
        #make the modified transformer encoder with prompt addition/removal and subtoken pooling functionality
        if self.config.model_source == 'flair':
            self.transformer_encoder_w_prompt = TransformerEncoderFlairPrompt(self.config)
        elif self.config.model_source == 'HF':
            self.transformer_encoder_w_prompt = TransformerEncoderHFPrompt(self.config)
        
        # hierarchical representation of tokens (Zaratiana et al, 2022)
        # https://arxiv.org/pdf/2203.14710.pdf
        #this just enriches the token reps using a single layer bilstm, maybe the reason he does this is due to the sw token to word token pooling that is done        
        #Nathan: all this is, is a single layer bilstm, where he takes the main output (last layer hidden states concatenated for fwd and rev)
        #thus the output is the same shape as the inptu and the bilstm is simply enriching the input bert embeddings by adding more sequential contextual information
        #I guess he feels that he needs it, one hting is that this is easy to turn on and off to see if it affects the otuptus in any way
        # I suspect he put this in to deal with the error he had int he flair bert implementation where he is averaging each layers output from bert instead of just taking the last layers output
        #thus I suspect this layer is not neccessary if bert was setup correctly, but it woudl be good to test it
        #additionlly this gives me an idea for using a bigru for incorporating gloabl context into the token reps => see my notes on this in te grapher folder!!
        self.rnn = LstmSeq2SeqEncoder(
            input_size      = config.hidden_size,
            hidden_size     = config.hidden_size // 2,
            num_layers      = 1,
            bidirectional   = True
        )


        #span width embeddings (in word widths)
        self.width_embeddings = nn.Embedding(config.max_span_width, config.width_embedding_size)
        #span representations
        #this forms the span reps from the token reps using the method defined by config.span_mode,
        self.span_rep_layer = SpanRepLayer(
            span_mode             = config.span_mode,
            max_seq_len           = config.max_seq_len,       #in word widths    
            max_span_width        = config.max_span_width,    #in word widths
            pooling               = config.subtoken_pooling,     #whether we are using pooling or not
            #the rest are in kwargs
            hidden_size           = config.hidden_size,
            width_embeddings      = self.width_embeddings,    #in word widths
            dropout               = config.dropout,
            ffn_ratio             = config.ffn_ratio, 
            use_span_pos_encoding = config.use_span_pos_encoding,    #whether to use span pos encoding in addition to full seq pos encoding
            cls_flag              = config.model_source == 'HF'    #whether we will have a cls token rep
        )

        #define the relation processor to process the raw x['relations'] data once we have our initial cand_span_ids
        self.rel_processor = RelationProcessor(self.config)
        
        #relation representation
        #this forms the rel reps from the cand_span_reps after the span reps have been filtered for the initial graph
        self.rel_rep_layer = RelationRepLayer(
            rel_mode    = config.rel_mode,    #what kind of rel_rep generation algo to use 
            hidden_size = config.hidden_size, 
            ffn_ratio   = config.ffn_ratio,
            dropout     = config.dropout,
            pooling     = config.subtoken_pooling,     #whether we are using pooling or not
        )

        # filtering layer for spans and relations
        self.span_filter_head = FilteringLayer(config.hidden_size)
        self.rel_filter_head = FilteringLayer(config.hidden_size)

        # graph embedder
        #this has code errors
        self.graph_embedder = GraphEmbedder(config.hidden_size)

        # transformer layer
        self.trans_layer = TransformerEncoderTorch(
            config.hidden_size,
            num_heads=config.num_heads,
            num_layers=config.num_transformer_layers
        )

        #keep_head
        #this seems to be a simple FFN then a binary classification head
        self.keep_head = FFNProjectionLayer(input_dim  = config.hidden_size, 
                                            ffn_ratio  = config.ffn_ratio, 
                                            output_dim = 1, 
                                            dropout    = 0.1)
        
        #scoring layers
        self.scorer_span = ScorerLayer(scoring_type = config.scorer,
                                       hidden_size  = config.hidden_size,
                                       dropout      = config.dropout)
        self.scorer_rel = ScorerLayer(scoring_type = config.scorer,
                                      hidden_size  = config.hidden_size,
                                      dropout      = config.dropout)

        self.init_weights()



    def init_weights(self):
        #you need to init weights here
        pass



    ###########################################################################
    #Base Functions, these were in basemodel before
    ###########################################################################
    def save_pretrained(self, save_directory: str):
        """Save the model parameters and config to the specified directory"""
        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), save_directory / "pytorch_model.bin")
        # Optionally save the configuration file
        save_to_json(save_directory / 'config.json')


    def load_pretrained(self, model_path):
        """Load model weights from the specified path"""
        state_dict = torch.load(model_path)
        self.load_state_dict(state_dict)
        print(f"Model loaded from {model_path}")


    def adjust_logits(self, logits, keep):
        """Adjust logits based on the keep tensor."""
        keep = torch.sigmoid(keep)
        keep = (keep > 0.5).unsqueeze(-1).float()
        adjusted_logits = logits + (1 - keep) * -1e9
        return adjusted_logits


    def predict(self, x, threshold=0.5, output_confidence=False):
        """Predict entities and relations."""
        out = self.forward(x, prediction_mode=True)

        # Adjust relation and entity logits
        out["span_logits"] = self.adjust_logits(out["span_logits"], out["keep_span"])
        out["rel_logits"] = self.adjust_logits(out["rel_logits"], out["keep_rel"])

        # Get entities and relations
        spans, rels         = er_decoder(x, 
                                         out["span_logits"], 
                                         out["rel_logits"], 
                                         out["topK_rel_idx"], 
                                         out["max_top_k"], 
                                         out["candidate_spans_idx"], 
                                         threshold=threshold, 
                                         output_confidence=output_confidence)
        return spans, rels


    def evaluate(self, eval_loader, threshold=0.5, batch_size=12, rel_types=None):
        self.eval()
        device = next(self.parameters()).device
        all_preds = []
        all_trues = []
        for x in eval_loader:
            for k, v in x.items():
                if isinstance(v, torch.Tensor):
                    x[k] = v.to(device)
            batch_predictions = self.predict(x, threshold)
            all_preds.extend(batch_predictions)
            all_trues.extend(get_relation_with_span(x))
        evaluator = Evaluator(all_trues, all_preds)
        out, f1 = evaluator.evaluate()
        return out, f1
    ###########################################################################
    ###########################################################################
    ###########################################################################



    def encode_train(self, x):
    #def compute_score_train(self, x):
        '''
        This just gets the device internally

        Inputs: x
        
        Operations:
        Passes the inputs (x) through the transformer encoder
        then through an LSTM layer to further enrich the embeddings
        then generates the span_reps

        Returns:
        token_reps/masks
        span_reps/masks
        sw_span_ids
        span_type_reps
        rel_type_reps
        w2sw_map if required
                
        '''
        # Process input
        #so here they add the unqiue entity and relation classes to the bert input and then separate the reps out after bert
        #so token_reps are the embeddings that come out of the encoder transfomer (either subword embeddings for no subtoken pooling, or word embeddings with subtoken pooling), span_type_reps are the reps for the entity classes, rel_type_reps are reps for relation classes etc..
        #NOTE: temp => we return both the w2sw_map list of dicts and the sw_span_ids tensor which is used in place of the span_ids tensor for the sw token aligned case.  I am deciding which one is more efficient, I suspect the sw_span_ids
        result = self.transformer_encoder_w_prompt(x, "train")
        token_reps     = result['token_reps']       #embeddings of the word/sw tokens
        token_masks    = result['token_masks']      #masks for the word/sw tokens
        span_type_reps = result['span_type_reps']   #embeddings for the span types
        rel_type_reps  = result['rel_type_reps']    #embeddings for the rel types
        cls_reps       = result['cls_reps']         #embeddings for the CLS sw token, only if we are using HF
        sw_span_ids    = result['sw_span_ids']      #tensor (batch, max_seq_len_batch*max_span_width, 2) => x['span_ids'] with values mapped using w2sw_map to the sw token start, end. Only if we are HF with no pooling.
        w2sw_map       = result['w2sw_map']         #w2sw mapping for the non-prompt and non special token word tokens to subword tokens. Only if we are HF with no pooling.
        '''
        ok so got here, we have 3 cases:
        1) flair => word emebddings, no cls_reps
        2) HF => word tokens, and have cls_reps
        This one you have to adjust for subword tokens
        3) HF => subword tokens, so need w2sw_map, sw_span_ids and have cls_reps
        '''
        #Enrich token_reps
        #enriches the (B,L,D) shaped token reps with a bilstm to (B, L, D) shaped token reps, 
        ###########################################################
        #My notes from graphER, you need to review whether this step has any benefit, esspecially for the no subtoken pooling case
        ###########################################################
        #not clear if this is really necessary, but may be to deal with the side effects of first sw token pooling to get word tokens, 
        #at least that is what they allude to in their associated paper https://arxiv.org/pdf/2203.14710
        #I suggest do not go to word tokens and then you do not need this stuff, however, I have an global context idea where this woudl be useful, see my comments in teh .txt file in the graphER folder
        token_reps = self.rnn(token_reps, 
                              token_masks) 

        #generates the span reps from the token reps and the span start/end idx and outputs a tensor of shape (B, L, max_span_width, D), i.e. the span reps are grouped by start idx (remember L*max_span_width == max_num_spans)
        span_reps = self.span_rep_layer(token_reps, 
                                        w_span_ids  = x['span_ids'], 
                                        span_masks  = x['span_masks'], 
                                        sw_span_ids = sw_span_ids, 
                                        cls_reps    = cls_reps,
                                        span_widths = x['span_ids'][:,:,1] - x['span_ids'][:,:,0])

        return dict(span_reps       = span_reps, 
                    token_reps      = token_reps, 
                    token_masks     = token_masks,
                    span_type_reps  = span_type_reps,    #will be None for use_prompt = false
                    rel_type_reps   = rel_type_reps,     #will be None for use_prompt = false
                    sw_span_ids     = sw_span_ids,   #will be None for pooling
                    w2sw_map        = w2sw_map)      #will be None for pooling



    @torch.no_grad()
    def encode_eval(self, x):
    #def compute_score_eval(self, x, device):
        '''
        Currently this is the same as the train case except for the @torch.no_grad
        I have not tested it yet, it may need to change, just start with this
        '''
        # Process input
        result = self.transformer_encoder_w_prompt(x, "eval")
        token_reps     = result['token_reps']       #embeddings of the word/sw tokens
        token_masks    = result['token_masks']      #masks for the word/sw tokens
        span_type_reps = result['span_type_reps']   #embeddings for the span types
        rel_type_reps  = result['rel_type_reps']    #embeddings for the rel types
        cls_reps       = result['cls_reps']         #embeddings for the CLS sw token, only if we are using HF
        sw_span_ids    = result['sw_span_ids']      #tensor (batch, max_seq_len_batch*max_span_width, 2) => x['span_ids'] with values mapped using w2sw_map to the sw token start, end. Only if we are HF with no pooling.
        w2sw_map       = result['w2sw_map']         #w2sw mapping for the non-prompt and non special token word tokens to subword tokens. Only if we are HF with no pooling.

        #Enrich token_reps
        token_reps = self.rnn(token_reps, 
                              token_masks) 

        #generates the span reps from the token reps and the span start/end idx and outputs a tensor of shape (B, L, max_span_width, D), i.e. the span reps are grouped by start idx (remember L*max_span_width == max_num_spans)
        span_reps = self.span_rep_layer(token_reps, 
                                        w_span_ids  = x['span_ids'], 
                                        span_masks   = x['span_masks'], 
                                        sw_span_ids = sw_span_ids, 
                                        cls_reps    = cls_reps,
                                        span_widths = x['span_ids'][:,:,1] - x['span_ids'][:,:,0])

        return dict(span_reps       = span_reps, 
                    token_reps      = token_reps, 
                    token_masks     = token_masks,
                    span_type_reps  = span_type_reps,    #will be None for use_prompt = false
                    rel_type_reps   = rel_type_reps,     #will be None for use_prompt = false
                    sw_span_ids     = sw_span_ids,   #will be None for pooling
                    w2sw_map        = w2sw_map)      #will be None for pooling


    ##################################################################################
    ##################################################################################
    ##################################################################################
    def forward(self, x, step=None, prediction_mode=False):
        '''
        x is a batch, which is a dict, with the keys being of various types as described below:
        x['tokens']     => list of ragged lists of strings => the raw word tokenized seq data as strings
        x['spans']      => list of ragged list of tuples => the positive cases for each obs 
        x['relations']  => list of ragged list of tuples => the positive cases for each obs
        x['orig_map']   => list of dicts for the mapping of the orig span idx to the span_ids dim 1 idx (for pos cases only, is needed in the model for rel tensor generation later)
        x['seq_length'] => tensor (batch) the length of tokens for each obs
        x['span_ids']   => tensor (batch, max_seq_len_batch*max_span_width, 2) => the span_ids truncated to the max_seq_len_batch * max_span_wdith
        x['span_masks'] => tensor (batch, max_seq_len_batch*max_span_width) => 1 for spans to be used (pos cases + selected neg cases), 0 for pad, invalid and unselected neg cases
        x['span_labels']=> tensor (batch, max_seq_len_batch*max_span_width) => 0 to num_span_types for valid cases, -1 for invalid and pad cases

        step will be current batch idx, i.e. we just set the total number of batch runs, say there are 1000 batches in the datset and we set pbar to 2200, then step will go from 0 to 2199, i.e. each batch will be run 2x and 200 will be 3x
        '''

        # compute span representation
        if prediction_mode:
            # Get the device of the model
            #device = next(self.parameters()).device
            result = self.encode_eval(x)
        else:
            #Compute scores for training
            result = self.encode_train(x)
        
        #read in data from results or x
        token_reps      = result['token_reps']          #(batch, max_seq_len_batch, hidden) => float, sw or w token aligned depedent pooling   
        token_masks     = result['token_masks']         #(batch, max_seq_len_batch) => bool, sw or w token aligned depedent pooling   
        w_span_ids      = x['span_ids'].clone()         #(batch, max_seq_len_batch * max_span_width, 2) => int, w aligned span_ids
        sw_span_ids     = result['sw_span_ids']         #(batch, max_seq_len_batch * max_span_width, 2) => int, sw aligned span_ids => None if pooling
        span_reps       = result['span_reps']           #(batch, max_seq_len_batch * max_span_width, hidden) => float
        span_masks      = x['span_masks'].clone()       #(batch, max_seq_len_batch * max_span_width) => bool
        span_labels      = x['span_labels'].clone()     #(batch, max_seq_len_batch * max_span_width) => int
        num_span_types  = len(self.config.span_types)   #scalar
        span_type_reps  = result['span_type_reps']      #(batch, num_span_types, hidden) => float, no mask needed
        num_rel_types   = len(self.config.rel_types)    #scalar
        rel_type_reps   = result['rel_type_reps']       #(batch, num_rel_types, hidden) => float, no mask needed
        '''
        NOTE: if use_prompt = false, the span_type_reps and rel_type_reps will be None here
        '''

        #get some dims
        batch, max_seq_len, hidden = token_reps.shape
        '''
        remember they denote dims as:
        B => batch (batch len)
        L => max_seq_len (for the batch, not self.config.max_seq_len)
        K => self.config.max_span_width
        D => hidden
        NOTE: num_spans = max_seq_len * self.config.max_span_width (inlcudes invalid spans etc...)
        graphER config has these params:
        max_top_k: 54
        add_top_k: 10
        '''

        #choose the top K spans per obs for the initial graph
        ###########################################################
        #Compute span filtering scores and binary CELoss for spans (only calcs for span within span_mask)
        #The filter score (batch, num_spans) float is a scale from -inf to +inf on the conf of the span being a pos case (> 0) or neg case (< 0)
        #the filter loss is an accumulated metric over all spans in all obs in the batch, so one scalar for the batch, indicating how well the model can detect that a span is postive case or negaitve case
        #determine the postive case forcing strategy first
        force_pos = self.config.span_force_pos
        if force_pos == True and (self.config.pos_force_step_limit != 'none' and self.config.pos_force_step_limit > step+1):
            force_pos = False
        #then run the scoring
        filter_score_span, filter_loss_span = self.span_filter_head(span_reps, 
                                                                    span_labels, 
                                                                    span_masks, 
                                                                    force_pos_cases=force_pos)
        #We now select the top K spans for the initial graph based on the span filter scores
        #first sort the filter_score_span tensor descending
        #NOTE [1] is to get the idx as opposed to the values as torch.sort returns a tuple (vals, idx)
        sorted_span_idx = torch.sort(filter_score_span, dim=-1, descending=True)[1]
        #next determine how many to shortlist (the K in top K), for longer sequences, it will be maxed at 64 spans per obs
        top_k_spans = min(max_seq_len, self.config.max_top_k_spans) + self.config.add_top_k_spans
        
        '''
        select the top_k spans from each obs and form the cand_span tensors (with the spans to use for the initial graph)
        This will create new tensors of length top_k_spans.shape[1] (dim 1) with the same order of span_idx as in teh top_k_spans tensor
        NOTE: these candidate tensors are smaller than the span_rep tensors, so it saves memory!!!, otherwise, I do not see the reason fro doing this, you coudl literally, just pass the span_idx_to_keeo
        '''
        #get the batch idx ad span idx for selection into the cand_span tensors
        batch_ind = torch.arange(batch).unsqueeze(-1)  # shape: (batch, 1)
        span_idx_to_keep = sorted_span_idx[:, :top_k_spans]
        #do the selection
        #get tensors for span_id (w_span_ids => map span boundaries to word tokens, sw_span_ids => map span boundaries to sw tokens, span_ids = w_span_ids for pooling or sw_span_ids)
        cand_w_span_ids = w_span_ids[batch_ind, span_idx_to_keep, :]  # shape: (batch, top_k_spans, 2)
        cand_span_ids = cand_w_span_ids
        if self.config.subtoken_pooling == 'none':
            cand_sw_span_ids = sw_span_ids[batch_ind, span_idx_to_keep, :]  # shape: (batch, top_k_spans, 2)
            cand_span_ids = cand_sw_span_ids
        #get the rps, masks and labels
        cand_span_reps = span_reps[batch_ind, span_idx_to_keep, :]  # shape: (batch, top_k_spans, hidden)
        cand_span_masks = span_masks[batch_ind, span_idx_to_keep]  # shape: (batch, top_k_spans)
        cand_span_labels = span_labels[batch_ind, span_idx_to_keep]  # shape: (batch, top_k_spans)


        '''
        Now process relations to find the subset to include in the initial graph
        - up to this point we only have the raw rel data => x['relations'], a list of dicts, we have no tensors
        - we need to form rel_reps, rel_labels, rel_ids, rel_masks
        NOTE: we do not use the cand prefix as this is the first time we make them....
        '''
        #get the rel labels from x['relations'] with dim 1 in same order as cand_span_id dim 1 expanded to top_k_spans**2 (this is why we need to limit top_k_spans to as low as possible)
        #NOTE: the rel_masks have the diagonal set to 0, i.e self relations masked out
        #this returns rel_labels and rel_masks, both of shape (batch, top_k_spans**2)
        rel_labels, rel_masks = self.rel_processor.get_cand_rel_tensors(x['relations'], 
                                                                        x['orig_map'],
                                                                        span_idx_to_keep,
                                                                        cand_span_ids, 
                                                                        cand_span_labels)

        '''
        Make the relation reps, has several options, based on config.rel_mode:
        1) no_context => graphER => juts concat the head and tail span reps
        2) between_context => concatenate the head and tail reps with between spans rep for context => to be coded
        3) window_context => concatenate the head and tail reps with windowed context reps, i.e. window context takes token reps in a window before and after each span and attention_pools, max_pool them => to be coded
        '''
        #output shape (batch, max_top_k**2, hidden)
        rel_reps = self.rel_rep_layer(cand_span_reps, 
                                      cand_span_ids, 
                                      token_reps, 
                                      token_masks)   

        #need test this rel_rep generation code do it for the no context and between context (w maxpooling) cases for now
        #need test this rel_rep generation code do it for the no context and between context (w maxpooling) cases for now
        #need test this rel_rep generation code do it for the no context and between context (w maxpooling) cases for now
        #need test this rel_rep generation code do it for the no context and between context (w maxpooling) cases for now
        #need test this rel_rep generation code do it for the no context and between context (w maxpooling) cases for now
        #need test this rel_rep generation code do it for the no context and between context (w maxpooling) cases for now
        #need test this rel_rep generation code do it for the no context and between context (w maxpooling) cases for now
        #need test this rel_rep generation code do it for the no context and between context (w maxpooling) cases for now
        #need test this rel_rep generation code do it for the no context and between context (w maxpooling) cases for now


        # Compute filtering scores for relations and sort them in descending order
        #Nathan:
        #this is identical code to the span case...
        # Compute filtering scores and CELoss for candidate relations from the candidate_relation_reps and the candidate_relation_labels
        #NOTE: unlike the spans, the rel_reps are already aligned with the candidate spans and so are the rel_labels (relation_classes)
        #The filter score per rel in each obs is a scaled from -inf to +inf on the conf of the rel being a positive rel (>0) or none_rel (<0)
        #the loss is an accumulated metric over all rels in all obs in the batch, so one scalar for the batch, indicating how well the model can detect that a rel is postive case (an relation) or negaitve case (none rel)
        #the binary classification head is trainable, so hopefully it gets better at determining if a rel is positive over time
        #NOTE: the structure of the binary classification head and the score and loss calc is identical to the span case
        filter_score_rel, filter_loss_rel = self.rel_filter_head(rel_reps, rel_labels)
        # Sort the filter scores for rels in descending order and just get the rel_idx 
        #NOTE [1] is to get the idx as opposed to the values as torch.sort returns a tuple
        sorted_idx_pair = torch.sort(filter_score_rel, dim=-1, descending=True)[1]

        # Embed candidate span representations
        #what he is doing here is generating the node specific identifier reps and node/edge discriminator reps and should be adding them to the raw span reps
        #!!!!!!the main issue is that he forgets to add the raw span reps to the node identifier and the node/edge dicriminator
        #the rel one is ok, as he deliberatly doesn't add the raw rel reps
        #################################################
        #some other weird things are that the node identifier will be also included in the rel matrix as self edges, surely this will eb an issue
        #I am not sold on the whole process here, it seems weird, but just go with it for now, the main error is forgetting to add the raw spans reps
        #but I reckon you could just come up with unique codes for each node as well as to discriminate between node and edge
        #I also think you coudl also add the raw rel reps, it makes sense to do so, esspecially if you add more context to the rel rep, whcih he is not doing as he is retarded
        #!!!!!!!!!!THIS NEEDS WORK!!!!!!!!!!!!!!!!!!!
        candidate_span_reps, cat_pair_rep = self.graph_embedder(candidate_span_reps)

        ###################################################
        #this is all jsut processing the rels, standard stuff
        ###################################################
        #this code basically selects out the max_top_k rel_reps and rel_labels, it does it in a not clear way though
        #it basically does this to recuce memory
        #Nathan: another needless list comrehension to make things annoying, just process the rel_reps and rel_labels, do not need a list comp
        #just make it clear like this, no abstracting away your dirty secrets mate.
        '''
        cat_par_rep = cat_pair_rep.view(B, max_top_k * max_top_k, -1)
        rel_idx_to_keep = sorted_idx_pair[:, :max_top_k]
        candidate_pair_rep = cat_pair_rep.gather(1, rel_idx_to_keep.unsqueeze(-1).expand(-1, -1, D))
        candidate_pair_label = rel_classes.gather(1, rel_idx_to_keep)
        '''
        # Define the elements to get candidates for
        elements = [cat_pair_rep.view(batch, top_k_spans * top_k_spans, -1), rel_labels.view(batch, top_k_spans * top_k_spans)]
        # Use a list comprehension to get the candidates for each element
        cand_pair_rep, cand_pair_label = [self.rel_processor.get_rel_candidates(sorted_idx_pair, element, topk=top_k_spans)[0] for element in elements]   #NAthan: ill be shape (B, max_top_k**2, D), (B, max_top_k**2)
        # Get the top K relation indices
        topK_rel_idx = sorted_idx_pair[:, :top_k_spans]
        # Mask the candidate pair labels using the condition mask and refine the relation representation
        #Nathan: do masking on teh remaining pairs using the same technique as used for the spans
        '''
        Remember this was done in the spans section:
        # Create a condition mask where the range of top K is greater than or equal to the top K lengths
        condition_mask = torch.arange(max_top_k, device=span_reps.device).unsqueeze(0) >= top_k_lengths.unsqueeze(-1)
        '''
        cand_pair_label.masked_fill_(condition_mask, -1)
        cand_pair_mask = cand_pair_label > -1
        ###################################################

        ###################################################
        # Concatenate span and relation representations
        ###################################################
        #Nathan: so he is merging the span_reps and rel_reps into one tensor, he is concat along dim 1, so span reps come first then rel reps, 
        #think of it like a sequence and each span or pair is a token, so he is concat the spans and pairs
        #so shape will be (B, max_top_k + max_top_k*max_top_k, D)
        #he is doing this to throw it into an attention block, the node speific identifiers and the node/edge discriminator shoudl help the attention to dscriminate 
        concat_span_pair = torch.cat((cand_span_reps, cand_pair_rep), dim=1)   #Nathan: shape (B, max_top_k + max_top_k**2, D)
        mask_span_pair = torch.cat((cand_span_masks, cand_pair_mask), dim=1)   #Nathan: shape (B, max_top_k + max_top_k**2)

        ###################################################
        # Apply transformer layer and keep_head
        ###################################################
        #this is a using the torch built in mha transformer encoder, mask is the key padding mask
        #seems to be setup properly, need to check, but looks ok, will be slooooow if you increase layers
        #shape will be same as input (B, max_top_k + max_top_k*max_top_k, D)
        out_trans = self.trans_layer(concat_span_pair, mask_span_pair)    #Nathan: shape will be (B, max_top_k + max_top_k**2, D)
        #the trans out reps go to a FFN then a binary classification head, i.e. last dim goes down to 1, then squeezed out
        #thuse we get one logit per node and edge that we can use to prune the graph
        #Nathan: keep_score will have shape (B, max_top_k + max_top_k*max_top_k)  => (max_top_k nodes + max_top_k^2 edges)
        keep_score = self.keep_head(out_trans).squeeze(-1)  # Shape: (B, max_top_k + max_top_k, 1)   #Nathan, this comment is def not correct!! shape will be (B, max_top_k + max_top_k**2, D)

        # Apply sigmoid function and squeeze the last dimension
        # keep_score = torch.sigmoid(keep_score).squeeze(-1)  # Shape: (B, max_top_k + max_top_k)  #Nathan: this comment is also wrong

        # Split keep_score into keep_ent and keep_rel
        #Nathan: this command is wrong and will not work
        #needs to be:
        #keep_span, keep_rel = keep_score.split([max_top_k, max_top_k**2], dim=1)
        keep_span, keep_rel = keep_score.split([top_k_spans, top_k_spans], dim=1)   #shoudl have out dims (B, max_top_k + max_top_k**2, D)

        """not use output from transformer layer for now"""
        # Split out_trans
        # candidate_span_rep, candidate_pair_rep = out_trans.split([max_top_k, max_top_k], dim=1)    #Nathan, this is wrong, out dims shoud be: (B, max_top_k, D), (B, max_top_k**2, D) 
        #NAthan: he has disabled this and also it is wrong again, probably why he disabled it!!!
        ####shoudl be: candidate_span_reps, candidate_pair_rep = out_trans.split([max_top_k, max_top_k**2], dim=1) 

        #got to here..................
        #got to here..................
        #got to here..................
        #got to here..................
        #at this point he should have the final span_reps and rel_reps and he should have pruned them with the keep_span and keep_rel outputs, but he has not
        #at this point he should have the final span_reps and rel_reps and he should have pruned them with the keep_span and keep_rel outputs, but he has not
        #at this point he should have the final span_reps and rel_reps and he should have pruned them with the keep_span and keep_rel outputs, but he has not

        # Compute scores for entities and relations
        #he has lots of scoring options, right now he is fixed on einsum, so the objects return:
        #return torch.einsum("bnd,btd->bnt", candidate_span_reps, span_type_reps)
        #return torch.einsum("bnd,btd->bnt", candidate_pair_rep, rel_type_reps)
        #Nathan: this einsum is effectively just a matmul on the last 2 dims, like this:
        #return torch.matmul(candidate_span_reps, span_type_reps.transpose(1, 2))
        #return torch.matmul(candidate_pair_rep, rel_type_reps.transpose(1, 2))
        #this the score is basically a logit/score for each entity type for each candidate span and similarly for the rels a score for each rel type for each candidate relation
        #remember span_type_reps is shape (B, num_entity_types, D) and rel_type_reps is shape (B, num_rel_types, D)
        scores_span = self.scorer_span(candidate_span_reps, span_type_reps)  # Shape: [B, max_top_k, num_entity_types]
        scores_rel = self.scorer_rel(candidate_pair_rep, rel_type_reps)  # Shape: [B, max_top_k**2, num_rel_types]

        if prediction_mode:
            return dict(span_logits          = scores_span,
                        relation_logits      = scores_rel,
                        keep_span            = keep_span,
                        keep_rel             = keep_rel,
                        candidate_spans_idx  = candidate_spans_idx,
                        candidate_pair_label = candidate_pair_label,
                        max_top_k            = max_top_k,
                        topK_rel_idx         = topK_rel_idx)

        #NAthan: for train mode, he just calculates a loss from all the heads and graph structure change etc and returns that
        # Compute losses for relation and entity classifiers
        rel_loss = compute_matching_loss(scores_rel, candidate_pair_label, rel_type_masks, num_rel)
        span_loss = compute_matching_loss(scores_span, candidate_span_labels, span_type_masks, num_span)

        # Concatenate labels for binary classification and compute binary classification loss
        span_rel_label = (torch.cat((candidate_span_labels, candidate_pair_label), dim=1) > 0).float()
        filter_loss = F.binary_cross_entropy_with_logits(keep_score, ent_rel_label, reduction='none')

        # Compute structure loss and total loss
        structure_loss = (filter_loss * mask_span_pair.float()).sum()
        total_loss = sum([filter_loss_span, filter_loss_rel, rel_loss, span_loss, structure_loss])

        return total_loss
