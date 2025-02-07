import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as init
from torch.profiler import record_function

from types import SimpleNamespace
import copy


###############################################
#custom imports
from .layers_transformer_encoder_flair import TransformerEncoderFlairPrompt
#from .layers_transformer_encoder_hf import TransformerEncoderHFPrompt
from .layers_transformer_encoder_hf_new import TransformerEncoderHFPrompt
from .layers_filtering import FilteringLayer
from .layers_other import LstmSeq2SeqEncoder, TransformerEncoderTorch, GraphEmbedder, OutputLayer
from .loss_functions import classification_loss
from .span_rep import SpanRepLayer
from .rel_processor import RelationProcessor
from .rel_rep import RelationRepLayer
from .utils import clear_gpu_tensors



class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        #create a locally mutable version of the config namespace
        self.config = SimpleNamespace(**config.get_data_copy())
        #special token definition
        self.config.s_token   = "<<S>>"
        self.config.r_token   = "<<R>>"
        self.config.sep_token = "<<SEP>>"
        ######################################
        self.num_limit = 6.5e4 if self.config.num_precision == 'half' else 1e9
        
        
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
            input_size      = self.config.hidden_size,
            hidden_size     = self.config.hidden_size // 2,
            num_layers      = 1,
            bidirectional   = True
        )


        #span width embeddings (in word widths)
        #NOTE: the number of embeddings needs to be max_span_width + 1 as the first one (idx 0) can be used for widths of length 0 (which are to be ignored, i.e. these are masked out spans anyway)
        #NOTE: do not need to specify the float dtype as we are using autocast
        self.width_embeddings = nn.Embedding(self.config.max_span_width + 1, 
                                             self.config.width_embedding_size) 
        #span representations
        #this forms the span reps from the token reps using the method defined by config.span_mode,
        self.span_rep_layer = SpanRepLayer(
            span_mode             = self.config.span_mode,
            max_seq_len           = self.config.max_seq_len,       #in word widths    
            max_span_width        = self.config.max_span_width,    #in word widths
            pooling               = self.config.subtoken_pooling,     #whether we are using pooling or not
            #rest are in kwargs
            hidden_size           = self.config.hidden_size,
            layer_type            = self.config.projection_layer_type,
            ffn_ratio             = self.config.ffn_ratio, 
            width_embeddings      = self.width_embeddings,    #in word widths
            dropout               = self.config.dropout,
            use_span_pos_encoding = self.config.use_span_pos_encoding,    #whether to use span pos encoding in addition to full seq pos encoding
            cls_flag              = self.config.model_source == 'HF'    #whether we will have a cls token rep
        )

        #filtering layer for spans
        self.span_filter_head = FilteringLayer(self.config.hidden_size, 
                                               num_limit = self.num_limit,
                                               dropout   = self.config.dropout if self.config.filter_dropout else None) 

        #define the relation processor to process the raw x['relations'] data once we have our initial cand_span_ids
        self.rel_processor = RelationProcessor(self.config)
        
        #relation representation
        #this forms the rel reps from the cand_span_reps after the span reps have been filtered for the initial graph
        self.rel_rep_layer = RelationRepLayer(
            rel_mode           = self.config.rel_mode,    #what kind of rel_rep generation algo to use 
            #rest are in kwargs
            hidden_size        = self.config.hidden_size, 
            layer_type         = self.config.projection_layer_type,
            ffn_ratio          = self.config.ffn_ratio,
            dropout            = self.config.dropout,
            pooling            = self.config.subtoken_pooling,     #whether we are using pooling or not
            no_context_rep     = self.config.rel_no_context_rep,   #how to handle edge case of no context tokens
            context_pooling    = self.config.rel_context_pooling,   #how to pool the context tokens
            window_size        = self.config.rel_window_size       
        )

        #filtering layer for relations
        self.rel_filter_head = FilteringLayer(self.config.hidden_size, 
                                              num_limit = self.num_limit,
                                              dropout   = self.config.dropout if self.config.filter_dropout else None) 


        # graph embedder
        self.graph_embedder = GraphEmbedder(self.config.hidden_size)
        
        # transformer layer
        self.trans_layer = TransformerEncoderTorch(self.config.hidden_size,
                                                   num_heads  = self.config.num_heads,
                                                   num_layers = self.config.num_transformer_layers)

        #used to calc the graph filter scores and graph loss
        self.graph_filter_head = FilteringLayer(self.config.hidden_size, 
                                                num_limit = self.num_limit,
                                                dropout   = self.config.dropout if self.config.filter_dropout else None) 
        
        #final output heads
        '''
        NOTE: 
        for unilabels the output dim will be num pos span/rel types + 1 for the none type
        for multilabels the output dim will be the num pos span/rel types with no none type
        '''
        self.output_head_span = OutputLayer(input_size  = self.config.hidden_size,
                                            output_size = self.config.num_span_types,
                                            dropout     = self.config.dropout,
                                            use_prompt  = self.config.use_prompt)
        
        self.output_head_rel = OutputLayer(input_size   = self.config.hidden_size,
                                           output_size  = self.config.num_rel_types,
                                           dropout      = self.config.dropout,
                                           use_prompt   = self.config.use_prompt)

        self.init_weights()



    def init_weights(self):
        #you need to init weights here
        pass


    def transformer_and_lstm_encoder(self, x):
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
                
        '''
        # Process input
        #so here they add the unqiue entity and relation classes to the bert input and then separate the reps out after bert
        #so token_reps are the embeddings that come out of the encoder transfomer (either subword embeddings for no subtoken pooling, or word embeddings with subtoken pooling), span_type_reps are the reps for the entity classes, rel_type_reps are reps for relation classes etc..
        with record_function("step_1: encoder transformer"):
            result = self.transformer_encoder_w_prompt(x)
        #read in the values from result
        token_reps     = result['token_reps']       #embeddings of the word/sw tokens
        token_masks    = result['token_masks']      #masks for the word/sw tokens
        span_type_reps = result['span_type_reps']   #embeddings for the span types
        rel_type_reps  = result['rel_type_reps']    #embeddings for the rel types
        cls_reps       = result['cls_reps']         #embeddings for the CLS sw token, only if we are using HF
        sw_span_ids    = result['sw_span_ids']      #tensor (batch, batch_max_seq_len*max_span_width, 2) => x['span_ids'] with values mapped using w2sw_map to the sw token start, end. Only if we are HF with no pooling.
        '''
        ok so got here, we have 3 cases:
        1) flair => word emebddings, no cls_reps
        2) HF => word tokens, and have cls_reps
        This one you have to adjust for subword tokens
        3) HF => subword tokens, so need sw_span_ids and have cls_reps
        '''
        #Enrich token_reps
        #enriches the (batch, seq_len, hidden) shaped token reps with a bilstm to (batch, seq_len, hidden) shaped token reps, 
        ###########################################################
        #My notes from graphER, you need to review whether this step has any benefit, esspecially for the no subtoken pooling case
        ###########################################################
        #not clear if this is really necessary, but may be to deal with the side effects of first sw token pooling to get word tokens, 
        #at least that is what they allude to in their associated paper https://arxiv.org/pdf/2203.14710
        #I suggest do not go to word tokens and then you do not need this stuff, however, I have an global context idea where this woudl be useful, see my comments in teh .txt file in the graphER folder
        '''
        I am not sure why they are doing this, the only benefit is that it doesn't have the length limit like the encoder
        REVIEW IF NEEDED AT ALL
        '''
        with record_function("step_1.5: lstm"):
            token_reps = self.rnn(token_reps, token_masks) 

        return dict(token_reps      = token_reps, 
                    token_masks     = token_masks,
                    span_type_reps  = span_type_reps,   #will be None for use_prompt = false
                    rel_type_reps   = rel_type_reps,    #will be None for use_prompt = false
                    sw_span_ids     = sw_span_ids,      #will be None for pooling
                    cls_reps        = cls_reps)         #if using HF




    def calc_top_k_spans(self):
        '''
        Determine a value for top_k_spans for this batch
        ---------------------
        SL = batch_match_seq_len    (could be self.config,max_seq_len, but coudl also be less, depends on the batch)
        W = max_span_width          (from self.config.max_span_width)
        W_mod = min(SL, W)
        ---------------------
        => num_spans = SL * W  (this includes those spans at that start in the seq len but end outside it)
        => num_spans_available = SL * W_mod - W_mod*(W_mod-1)/2    (start/end in the batch_max_seq_len, potentially not in the actual seq len of that obs, but inside the batch_max_seq_len)
        '''
        #first determine the number of spans inside the batch_max_seq_len, these are spans that are possible to shortlist
        #NOTE: that some of them could be maksed out, but at least they exist in the token_reps tensor
        S = self.batch_max_seq_len
        W_mod = min(S, self.config.max_span_width)
        num_spans_available = S * W_mod - W_mod*(W_mod-1)/2
        top_k_spans = min(num_spans_available, self.config.max_top_k_spans)
        return int(top_k_spans)
        




    def calc_top_k_rels(self, rel_scores):
        """
        Dynamically determines the number of top relationships to include in the graph based on a percentile threshold of relationship scores.

        This function calculates the threshold score at a specified percentile of the relationship scores provided. It then counts how many relationships exceed this threshold and limits the count based on the maximum number allowed by the configuration. 
        The function ensures that the number of relationships considered does not exceed the actual number of relationships available.

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
        if self.config.num_precision == 'half':
            #quantile doesn't support half precision
            rel_score_thd = torch.quantile(rel_scores.to(dtype=torch.float32), self.config.rel_score_percentile/100)
        else:
            rel_score_thd = torch.quantile(rel_scores, self.config.rel_score_percentile/100)
        valid_rels_mask = rel_scores >= rel_score_thd
        valid_rels_count = valid_rels_mask.sum().item()
        top_k_rels = min(valid_rels_count, self.config.max_top_k_rels, rel_scores.shape[1])
        return int(top_k_rels)




    def prune_spans(self, 
                    filter_score_span, 
                    top_k_spans):
        '''
        select the top K spans for the initial graph based on the span filter scores
        '''
        #first sort the filter_score_span tensor descending
        #NOTE [1] is to get the idx as opposed to the values as torch.sort returns a tuple (vals, idx)
        sorted_span_idx = torch.sort(filter_score_span, dim=-1, descending=True)[1]
        
        '''
        select the top_k spans from each obs and form the cand_span tensors (with the spans to use for the initial graph)
        This will create new tensors of length top_k_spans.shape[1] (dim 1) with the same order of span_idx as in the top_k_spans tensor
        NOTE: these candidate tensors are smaller than the span_rep tensors, so it saves memory!!!, otherwise, I do not see the reason fro doing this, you coudl literally, just pass the span_idx_to_keep
        '''
        #get the span idx for selection into the cand_span tensors
        span_idx_to_keep = sorted_span_idx[:, :top_k_spans]    #(batch, top_k_spans)

        #extra processing
        #make the mapping from span_ids idx to cand_span_ids idx using the span_idx_to_keep shortlist.  This needs to be returned
        span_to_cand_span_map = torch.full((self.batch, self.num_spans), -1, dtype=torch.long, device=self.device)
        cand_span_indices = torch.arange(top_k_spans, device=self.device).expand(self.batch, -1)  # Shape (batch, top_k_spans)
        span_to_cand_span_map[self.batch_ids, span_idx_to_keep] = cand_span_indices   #shape (batch, num_spans)
        
        return span_idx_to_keep, span_to_cand_span_map



    def prune_rels(self, filter_score_rel, top_k_rels):
        '''
        filter the rels down to top_k_rels with the rel filter scores
        '''
        #Sort the filter scores for rels in descending order and just get the rel_idx 
        #NOTE [1] is to get the idx as opposed to the values as torch.sort returns a tuple
        sorted_rel_idx = torch.sort(filter_score_rel, dim=-1, descending=True)[1]    #(batch, top_k_spans**2)
        #do the relation pruning
        #Select the top_k relationships for processing
        rel_idx_to_keep = sorted_rel_idx[:, :top_k_rels]   #(batch, top_k_rels)

        return rel_idx_to_keep





    def binarize_labels(self, labels):
        '''
        Make the binary int labels from labels (handles the multilabel and unilabel cases)
        labels_b will be shape (batch, num_reps) int with values of 0 or 1
        NOTE: returns None if labels is None
        '''
        if labels is None:
            return None
        
        if len(labels.shape) == 3:  # Multilabel case, all zero vector is a neg case, otherwise its a pos case
            return labels.any(dim=-1).to(torch.long)  # Aggregate multilabels to binary
        else:  # Unilabel case, any label > 0 is a pos case otherwise a neg case
            return (labels > 0).to(torch.long)


    def set_force_pos(self, force_pos, step):
        if force_pos == True and (self.config.pos_force_step_limit != 'none' and self.config.pos_force_step_limit > step+1):
            force_pos = False
        return force_pos



    def run_output_heads(self, node_reps, span_type_reps, edge_reps, rel_type_reps):
        '''
        run the reps through the respective output classification head and return the logits
        '''
        logits_span = self.output_head_span(node_reps, span_type_reps)  # Shape: (batch, top_k_spans, num_span_types)
        logits_rel = self.output_head_rel(edge_reps, rel_type_reps)   # Shape: (batch, top_k_rels, num_rel_types)
        return logits_span, logits_rel



    def prune_graph_logits(self, logits_span, filter_score_nodes, logits_rel, filter_score_edges):
        '''
        NOTE: the pruning is done by biasing the logits to very negative (-1e9 or -6.5e4) for nodes/edges that we want to prune out
              this effectively masks out these nodes/edges from all downstream loss/prediction functions
        '''
        logits_span = self.graph_filter_head.apply_filter_scores_to_logits(logits_span, filter_score_nodes, self.config.node_keep_thd)
        logits_rel = self.graph_filter_head.apply_filter_scores_to_logits(logits_rel, filter_score_edges, self.config.edge_keep_thd)
        return logits_span, logits_rel




    ##################################################################################
    ##################################################################################
    ##################################################################################
    def forward(self, x, step=None):
        '''
        x is a batch, which is a dict, with the keys being of various types as described below:
        x['tokens']     => list of ragged lists of strings => the raw word tokenized seq data as strings
        x['seq_length'] => tensor (batch) the length of tokens for each obs
        x['span_ids']   => tensor (batch, batch_max_seq_len*max_span_width, 2) => the span_ids truncated to the batch_max_seq_len * max_span_wdith
        x['span_masks'] => tensor (batch, batch_max_seq_len*max_span_width) => 1 for spans to be used (pos cases + selected neg cases), 0 for pad, invalid and unselected neg cases  => if no labels, will be all 1 for valid/non pad spans, no neg sampling
        x['span_labels']=> tensor (batch, batch_max_seq_len*max_span_width) int for unilabels.  (batch, batch_max_seq_len*max_span_width, num_span_types) bool for multilabels.  Padded with 0.
        x['spans']      => list of ragged list of tuples => the positive cases for each obs  => list of empty lists if no labels
        x['relations']  => list of ragged list of tuples => the positive cases for each obs  => list of empty lists if no labels
        x['orig_map']   => list of dicts for the mapping of the orig span idx to the span_ids dim 1 idx (for pos cases only, is needed in the model for rel tensor generation later) => list of empty dicts if no labels

        step will be current batch idx, i.e. we just set the total number of batch runs, say there are 1000 batches in the datset and we set pbar to 2200, then step will go from 0 to 2199, i.e. each batch will be run 2x and 200 will be 3x
        
        NOTE: if self.config.run_type == 'predict', there are no labels so the following differences are:
        - x['spans'] => None
        - x['relations'] => None
        - x['orig_map'] => None
        - x['span_labels'] => None
        '''
        #determine if we have labels and thus need to calc loss
        has_labels = self.config.run_type == 'train'
         
        #Run the transformer encoder and lstm encoder layers
        result = self.transformer_and_lstm_encoder(x)
        #read in data from results or x
        token_reps      = result['token_reps']          #(batch, batch_max_seq_len, hidden) => float, sw or w token aligned depedent pooling   
        token_masks     = result['token_masks']         #(batch, batch_max_seq_len) => bool, sw or w token aligned depedent pooling   
        w_span_ids      = x['span_ids']                 #(batch, batch_max_seq_len * max_span_width, 2) => int, w aligned span_ids
        sw_span_ids     = result['sw_span_ids']         #(batch, batch_max_seq_len * max_span_width, 2) => int, sw aligned span_ids => None if pooling
        span_masks      = x['span_masks']               #(batch, batch_max_seq_len * max_span_width) => bool
        span_labels     = x['span_labels']              #(batch, batch_max_seq_len * max_span_width) int or (batch, batch_max_seq_len * max_span_width, num_span_types) bool
        num_span_types  = self.config.num_span_types    #scalar, includes the none type (idx 0) for unilabel and only pos types for multilabel
        span_type_reps  = result['span_type_reps']      #(batch, num_span_types, hidden) => float, no mask needed
        num_rel_types   = self.config.num_rel_types     #scalar, includes the none type (idx 0) for unilabel and only pos types for multilabel
        rel_type_reps   = result['rel_type_reps']       #(batch, num_rel_types, hidden) => float, no mask needed
        cls_reps        = result['cls_reps']            #will be None for flair
        #NOTE: if use_prompt = false, the span_type_reps and rel_type_reps will be None here
        #NOTE: for prompting, if unilabels the type reps will include the none type rep, if multilabels the type reps will only be for the positive types
        
        #######################################################
        # SPANS ###############################################
        #######################################################
        #generate the span reps from the token reps and the span start/end idx and outputs a tensor of shape (batch, num_spans, hidden), 
        #i.e. the span reps are grouped by start idx
        #NOTE: for all further analyses => num_spans is the total num spans including those that end outside the max seq len, they are just masked out
        #span_reps shape (batch, batch_max_seq_len * max_span_width, hidden) => float
        with record_function("step_2: make span reps"):
            span_reps = self.span_rep_layer(token_reps, 
                                            w_span_ids  = w_span_ids, 
                                            span_masks  = span_masks, 
                                            sw_span_ids = sw_span_ids, 
                                            cls_reps    = cls_reps,
                                            span_widths = w_span_ids[:,:,1] - w_span_ids[:,:,0],
                                            neg_limit   = -self.num_limit)

        #get some dims add to self for convenience
        self.batch, self.batch_max_seq_len, _ = token_reps.shape
        self.num_spans = span_reps.shape[1]
        self.device = token_reps.device
        self.batch_ids = torch.arange(self.batch, dtype=torch.long, device=self.device).unsqueeze(-1)  # shape: (batch, 1)
        
        #choose the top K spans per obs for the initial graph
        ###########################################################
        #Compute span filtering scores and binary CELoss for spans (only calcs for span within span_mask)
        #The filter score (batch, num_spans) float is a scale from -inf to +inf on the conf of the span being a pos case (> 0) or neg case (< 0)
        #the filter loss is an accumulated metric over all spans in all obs in the batch, so one scalar for the batch, indicating how well the model can detect that a span is postive case or negaitve case
        
        #get filter_score and filter_loss
        #NOTE: 
        #- we send in the span_masks here to force all masked out span scores to -inf
        #- we also force all pos cases to +inf to ensure they are first (if force_poos)
        #However, there could be caess where the number of pos cases and selected_neg cases is less than the top_k_spans,
        #So we could have a few -inf scores for masked out spans...
        #This is ok, just be aware, we also generate the cand_span_mask below to indicate this 
        #output will be shapes:  (batch, num_spans), scalar 
        #loss will be 0 for has_labels = False
        with record_function("step_3: get span filter scores and loss"):
            filter_score_span, filter_loss_span = self.span_filter_head(span_reps, 
                                                                        span_masks, 
                                                                        self.binarize_labels(span_labels), 
                                                                        force_pos_cases = self.set_force_pos(self.config.span_force_pos, step),
                                                                        reduction = 'sum')           
        with record_function("step_4: calc top_k_spans"):
            #get top_k_spans
            top_k_spans = self.calc_top_k_spans()
        with record_function("step_5: get span ids to keep"):
            #prune the spans to top_k_spans
            span_idx_to_keep, span_to_cand_span_map = self.prune_spans(filter_score_span, top_k_spans) 
        with record_function("step_6: prune spans"):
            #filter the reps, masks and labels
            cand_span_reps = span_reps[self.batch_ids, span_idx_to_keep].clone()   # shape: (batch, top_k_spans, hidden)
            cand_span_masks = span_masks[self.batch_ids, span_idx_to_keep].clone()    # shape: (batch, top_k_spans)
            cand_span_labels = span_labels[self.batch_ids, span_idx_to_keep].clone() if has_labels else None  # shape: (batch, top_k_spans) unilabels or (batch, top_k_spans, num_span_types) multilabels
            cand_w_span_ids = w_span_ids[self.batch_ids, span_idx_to_keep, :].clone()  # shape: (batch, top_k_spans, 2)
            #set cand_span_ids to either cand_w_span_ids or cand_sw_span_ids
            cand_span_ids = cand_w_span_ids
            if self.config.subtoken_pooling == 'none':
                cand_sw_span_ids = sw_span_ids[self.batch_ids, span_idx_to_keep, :].clone()  # shape: (batch, top_k_spans, 2)
                cand_span_ids = cand_sw_span_ids


        ###############################################################
        # RELATIONS ###################################################
        ###############################################################
        '''
        Now that we have pruned the spans to a manageable level, we can generate all possible rels from top_k_spans, i.e. top_k_spans**2 rel candidates
        Then we will align labels/masks for them and masks, make reps and then filter them to find the best topk_rels to prune the rels to
        This requires a lot of procssing intra-model, i.e. 
        - up to this point we only have the raw rel annotations => x['relations'], a list of dicts, we have no tensors
        - we need to form rel_reps, rel_labels, rel_ids, rel_masks
        ##############################################################
        ReadMe on span teacher forcing and dealing with the downstream effect on lost rels when we turn it off
        ##############################################################
        For this model there are 2 main stages, the span filtering (shortlisting) stage and then the results of this shortlist are used to form the relations
        which are then filtered.  Then both shortlisted spans and rels are classified where it is guaranteed that head/tails spans in the shortlisted rels are always in the shortlisted spans 
        as the rels were derived from the span shortlist.
        However, as you can see, because the rels are dependent on the span shortlist results, we have a pipeline effect setup.
        This means that if an annotated span misses the span shortlist, then the corresponding relation containing that span as head or tail can never be formed, thus the model
        never has the opportunity to even predict on this rel.  Call this issue the missed span => lost relation problem or even just the lost relation problem.  We also have a missed relation problem 
        where the model has a positive relation rep to predict on, but just misclassifies it as a negative case (this could be due to rel filtering, i.e. it missed the rel shortlist, or due to final 
        classification head classificiation, where it was in the shortlist but classified as not a rel).  This is less of a problem and inherantly handled by the model loss (either filter loss or classification loss)
        The same goes for spans in that is a positive annotated span is filterd out then it misses the span shortlist (this is handled by the span filtering loss) and if it makes the shortlist but is misclassified as not a span
        it is handled by the span classification loss.  But it is this pipeline effect from missed span to lost relation that is NOT handled by the model loss structure and that is what needs to be added for the non-teacher 
        forced spans mode of operation.  In that case we need a lost_rel loss which is an extra penatly applied to the model when a missed span (incorrectly filtered out span) results in a lost relation (positive annotated relaation
        that never makes it to the pre-filtered relation reps)         
        #######################################################
        In summary, there are 6 stages where we need loss in this model, 5 from heads and 1 from a non-head:
        1) form span reps => span filtering head => if not span teacher forcing => results in missed spans => handled by span filter loss
        2) rel generation function => if not span teacher forcing => we can get lost rels where a positively annotated rel is never formed from the shortlisted spans as the positively annotated span was missed by the previous filter stage => this needs to be handled by a lost rel loss
        3) form rel reps => rel filtering head => if not rel teacher forcing => results in missed rels => handled by rel filter loss.  
        4) <optional> form graph (node/edge) reps => graph filtering head => if not graph teacher forcing => results in missed spans/rels => handled by the graph filter loss
        5) span classification head => never teacher forced => results in misclassified spans => handled by the span classification loss
        6) rel classification head => never teacher forced => results in misclassified rels => handled by the rek classification loss
        
        In the case we have to go to a more complex model pipeline (11 stages as opposed to 6).  We apply the same techniques, the pipeline just gets more complex
        We would have trigger annotations (like spans), arg annotations (like spans), arg-trigger annotations (like rels), trigger-trigger (proxy for event-event) annotations (like rels)
        NOTE: triggers are proxies for event/states, 1 per event/state, can have zero or more args per trigger, an arg can be related to more than one trigger but rarely
        NOTE: we thus form the event reps from merging the trigger rep with the reps for all associated args along with any relvant context, somewhat more complex than the simple span-rel scenario, but not so bad
        NOTE: we would most likely make 5 sets of labels => trigger labels (from triggers), arg labels from (args), arg-trigger labels (from trigger, args, arg-triggers), event labels (from triggers, args, arg-triggers *different to single arg-triggers), event-event labels (triggers, args, arg-triggers, trigger-triggers)
        One good way would be to concat the trigger rep with the attention pooled arg reps with the attention pooled context reps.  With the attention pooling using the trigger rep as query.
        I would just go for maxpooling to start with though
        The pipeline architecture....
        1) form trigger reps => trigger filtering head => if not trigger teacher forcing => results in missed triggers => handled by trigger filter loss
        2) form arg reps => arg filtering head => if not arg teacher forcing => results in missed args => handled by arg filter loss.  
        2* => try the simple way first where the arg reps are not dependent on the trigger rep (fast), but can also try a modified version where we form the arg-trigger reps for all combos of span to trigger where we only take
        canidate spans within a window of the trigger, say 30 tokens before and 30 tokens after and the cand spans can not overlap the trigger span.  This would expand the number of arg spans to check to something like max_span_width * (window * 2 - 1) * num_shortlisted triggers
        This will not work as even though we are refining the set to check, it still suffers geometric expansion (so would only work if the trigger shortlist was < 10).
        Alternately, you could use prompting, so you put in the trigger and arg types with tokens as a prompt (like grpahER) and generate the reps for these in bert, then mix these prompt_trigger/arg reps with the span reps to make the candidate 
        trigger reps and arg reps to send to each of the trigger and arg filtering head respectively, i.e. trigger rep for one span = concat(prompt_trigger_rep, maxpool(token reps in that span), arg rep for the same span = concat(prompt_arg_rep, maxpool(token reps in that span)
        This would be a lot faster than the first idea... and the args are not really dependent on the triggers so we do not get the lost arg problem
        3) arg-trigger generation function => if not arg or trigger tf => results in lost arg-triggers => handled by lost arg-trigger loss from missing arg + lost arg-trigger loss from missing trigger (dependent on trigger or arg teacher forcing being off), NOTE: missed args are not as critical as missed triggers so the loss penalty would be potentially less for missed args
        4) form arg-trigger reps => if not arg-trigger tf => results in missed arg-triggers => handled by arg-trigger filter
        5) <special> event generation ([args]-trigger) function => if not arg-trigger teacher forcing => results in lost events from missing arg-triggers from missed trigger NOTE: as events are a grouping of arg-triggers, we do not create more loss, we already have accounted for the loss in the arg-triggers from missed triggers
        6) form event reps ([args]-trigger) => event filtering head => if not event teacher forcing => results in missed event => handled by event filter loss.  
        7) event-event generation function => if not event teacher forcing => we can get lost event-event rels => handled by lost-event-event loss
        8) form event-event reps => event-event filtering head => if not event-event teacher forcing => results in missed event-event => handled by event-event filter loss.  
        9) <optional> form event graph (node/edge) reps => graph filtering head => if not graph teacher forcing => results in missed event/event-events => handled by the graph filter loss
        10) event classification head => never teacher forced => results in misclassified events => handled by the event classification loss
        11) event-event classification head => never teacher forced => results in misclassified event-events => handled by the event-event classification loss
        Summary, we would have 10 losses:
        - 3 lost loses (lost arg-trigger from missing arg, lost arg-trigger from missing trigger, lost event-event from missing trigger [if missing arg, is ok as long as we have the trigger])
        - 6 filter losses (trigger filter, arg filter, arg-trigger filter, event filter, event-event filter, event graph filter)
        NOTE: it would be nice to initially just do trigger filtering, then use that shortlist to form arg-trigger reps (even simply like concat maxpool(trigger token reps) + maxpool(arg token reps) + maxpool(context token reps))
        where we just go through all possible spans for the arg (except for the spans that overlap the trigger span.) 
        '''
        #get the rel labels from x['relations'] with dim 1 in same order as cand_span_id dim 1 expanded to top_k_spans**2 (this is why we need to limit top_k_spans to as low as possible)
        #NOTE: the rel_masks have the diagonal set to 0, i.e self relations masked out
        #this returns rel_labels of shape (batch, top_k_spans**2) int for unilabel or (batch, top_k_spans**2, num span types) bool for multilabel => None if not has_labels
        #as well as rel_masks, rel_ids of shape (batch, top_k_spans**2), (batch, top_k_spans**2, 2) respectively
        #if has_labels, it also returns the lost_rel_cnts tensor of shape (batch) which is one int per batch obs indicating the number fo rels that had annotations but did not make it to get rel_ids as they were filtered out byt the span filtering
        #we will add a lost rel penalty loss for this
        with record_function("step_7: make rel tensors from pruned spans, except the rel reps"):
            rel_ids, rel_masks, rel_labels, lost_rel_counts, lost_rel_penalty = self.rel_processor.get_cand_rel_tensors(cand_span_masks,       #the span mask for each selected span  (batch, top_k_spans)
                                                                                                                        x['relations'],        #the raw relations data.  NOTE: will be None for no labels
                                                                                                                        x['orig_map'],         #maps raw span idx to span_ids idx.  NOTE: will be None for no labels  
                                                                                                                        span_to_cand_span_map) #the mapping from span_ids to cand_span_ids for each batch item (batch, num_spans)

        '''
        Make the relation reps, has several options, based on config.rel_mode:
        1) no_context => graphER => juts concat the head and tail span reps
        2) between_context => concatenate the head and tail reps with between spans rep for context => to be coded
        3) window_context => concatenate the head and tail reps with windowed context reps, i.e. window context takes token reps in a window before and after each span and attention_pools, max_pool them => to be coded
        4+) working on more options, see code for details
        '''
        with record_function("step_8: make rel reps"):
            #output shape (batch, top_k_spans**2, hidden)
            rel_reps = self.rel_rep_layer(cand_span_reps = cand_span_reps, 
                                          cand_span_ids  = cand_span_ids, 
                                          token_reps     = token_reps, 
                                          token_masks    = token_masks,
                                          rel_masks      = rel_masks,
                                          neg_limit      = -self.num_limit)

        #Compute filtering scores for relations and sort them in descending order
        ###############################################################
        #Compute filtering scores and CELoss for candidate relations from the rel_reps and the rel_labels
        #NOTE: unlike the spans, the rel_reps are already aligned with the candidate spans and so are the rel_labels (relation_classes)
        #The filter score per rel in each obs is a scaled from -inf to +inf on the conf of the rel being a positive rel (>0) or none_rel (<0)
        #the loss is an accumulated metric over all rels in all obs in the batch, so one scalar for the batch, indicating how well the model can detect that a rel is postive case (an relation) or negaitve case (none rel)
        #the binary classification head is trainable, so hopefully it gets better at determining if a rel is positive over time
        #NOTE: the structure of the binary classification head and the score and loss calc is identical to the span case
        ########################################################
        #run the filter
        #output will be shapes:  (batch, top_k_spans**2), scalar
        with record_function("step_9: make rel filter scores and loss"):
            filter_score_rel, filter_loss_rel = self.rel_filter_head(rel_reps, 
                                                                    rel_masks,
                                                                    self.binarize_labels(rel_labels),
                                                                    force_pos_cases = self.set_force_pos(self.config.rel_force_pos, step),
                                                                    reduction = 'sum')           
        with record_function("step_10: calc the top_k_rels"):
            #Calculate the number of rels to shortlist
            top_k_rels = self.calc_top_k_rels(filter_score_rel)   #returns a scalar which should be << top_k_spans**2
        with record_function("step_11: calc the rel_idx_to_keep"):
            #prune the rels
            rel_idx_to_keep = self.prune_rels(filter_score_rel, top_k_rels)
        with record_function("step_12: prune the rels"):
            cand_rel_reps   = rel_reps[self.batch_ids, rel_idx_to_keep].clone()    #(batch, top_k_rels, hidden)  float
            cand_rel_masks  = rel_masks[self.batch_ids, rel_idx_to_keep].clone()      #(batch, top_k_rels)  bool
            cand_rel_ids    = rel_ids[self.batch_ids, rel_idx_to_keep].clone()     #(batch, top_k_rels, 2) int
            cand_rel_labels = rel_labels[self.batch_ids, rel_idx_to_keep].clone() if has_labels else None     #(batch, top_k_rels) int for unilabel or (batch, top_k_rels, num rel types) int for multilabel 
        with record_function("step_13: clear unneeded tensors"):
            #clear rel_reps
            #NOTE: if you want ot take this out, just remove the .clone() for the cand_span/rel commands also
            #NOTE: just del the tensors you do not need for now, it is fast
            clear_gpu_tensors([span_reps, span_masks, w_span_ids, sw_span_ids, span_labels, rel_reps, rel_masks, rel_ids, rel_labels], 
                               gc_collect=True if (step + 1) % self.config.clear_tensor_steps == 0 else False,    #slows it down if true
                               clear_cache=True if (step + 1) % self.config.clear_tensor_steps == 0 else False)    #slows it down if true

        if self.config.use_graph:
            with record_function("step_14.1: run graph embedder"):
                #Generate the node and edge reps
                #this basically superimposes a node identifier to the span_reps and an edge identifier to the rel_reps
                #i.e. the id is added by element wise addition to the reps, the same way as pos encoding is added
                #this is preparing the reps for graph attention
                #node reps with be shape (batch, top_k_spans, hidden)
                #edge reps with be shape (batch, tp_k_rels, hidden)
                node_reps, edge_reps = self.graph_embedder(cand_span_reps, cand_rel_reps, cand_span_masks, cand_rel_masks)

            with record_function("step_14.2: make graph reps and labels"):
                #merge the node and edge reps into one tensor so we can run it through a attention block
                #think of it like a sequence and each span or pair is a token, so he is concat the spans and pairs
                #so shape will be (batch, top_k_spans + top_k_rels, hidden)
                graph_reps = torch.cat((node_reps, edge_reps), dim=1)   #shape (batch, top_k_spans + top_k_rels, hidden)   float
                #merge masks
                graph_masks = torch.cat((cand_span_masks, cand_rel_masks), dim=1)   #(batch, top_k_spans + top_k_rels)   bool
                #binarize and merge labels as this is just used for graph pruning
                graph_labels_b = torch.cat((self.binarize_labels(cand_span_labels), 
                                            self.binarize_labels(cand_rel_labels)), dim=1) if has_labels else None   #(batch, top_k_spans + top_k_rels) int (0,1) for unilabel/multilabel as they have been binarized

                ###################################################
                #Apply transformer layer and keep_head
                #Q: why not use a GAT => beacuse the GAT doesn't enrich the edge reps, only the node reps, even if we get it to take in the edge reps, it is garbage
                ###################################################
                #this is a using the torch built in mha transformer encoder, mask is the key padding mask
                #shape out will be same as input reps (batch, top_k_spans + top_k_rels, hidden)
                '''
                This could be a key sticking point.  
                GraphER is trying to use the torch implementation of an transformer encoder, this could be slow and is not pretrained.
                I suggest investigating options here:
                1) potentially simpifying to just a single layer
                2) use a bert or big bird instance
                '''
            with record_function("step_14.3: run graph transformer"):
                graph_reps = self.trans_layer(graph_reps, graph_masks)
                
            with record_function("step_14.4: calc graph filter score and loss"):
                #run graph reps through a binary classification head
                #output will be shapes:  (batch, top_k_spans + top_k_rels), scalar (None if has_labels is False)
                filter_score_graph, filter_loss_graph = self.graph_filter_head(graph_reps, 
                                                                            graph_masks,
                                                                            graph_labels_b,    #will be None if has_labels is False
                                                                            force_pos_cases = self.set_force_pos(self.config.graph_force_pos, step),
                                                                            reduction = 'none')           

            with record_function("step_14.5: split graph reps/scores back to node and edge"):
                #Split the scores and reps back to nodes(spans) and edges(rels)
                filter_score_nodes, filter_score_edges = filter_score_graph.split([top_k_spans, top_k_rels], dim=1)   #(batch, top_k_spans + top_k_rels) => (batch, top_k_spans), (batch, top_k_rels)
                node_reps, edge_reps = graph_reps.split([top_k_spans, top_k_rels], dim=1)                             #(batch, top_k_spans + top_k_rels, hidden) => (batch, top_k_spans, hidden), (batch, top_k_rels, hidden)

                #set the node and edge reps back to the pre-graph transformer cand_span_reps and cand_rel_reps if use_graph_reps is False
                #I do not know why this woudl be beneficial, but potentially the graph transformer doesn't help the reps and is only good for pruning
                #this is what graphER was doing, worth doing ablation tests on
                if not self.config.use_graph_reps:
                    node_reps, edge_reps = cand_span_reps, cand_rel_reps

                '''
                at this point we have the final output for spans(nodes) and rels(edges):
                spans
                - node_reps, 
                - filter_score_nodes,
                - cand_span_masks, 
                - cand_span_labels, 
                - cand_span_ids
                - cand_w_span_ids
                rels
                - edge_reps, 
                - filter_score_edges,
                - cand_rel_masks, 
                - cand_rel_labels, 
                - cand_rel_ids
                - lost_rel_counts
                '''

                ############################################################################
                ############################################################################
                ############################################################################
                #Output Heads
                ############################################################################
                ############################################################################
                ############################################################################
                '''
                make the output logits for spans and rels
                NOTE: for the prompting case the output head is not trainable, just an einsum similarity function
                    but it outputs logits of the same shape as the trainable output head used in the no prompting case
                #####################################################################################
                NOTE: num_span_types and num_rel_types are the number of neg/pos types for unilabel and pos types only for multilabel
                NOTE: the span and rel type reps will both be None for no prompting
                #####################################################################################
                '''
            with record_function("step_15.1: run output heads"):
                #logits span will be (batch, top_k_spans, num_span_types), rels will be (batch, top_k_rels, num_rel_types)
                logits_span, logits_rel = self.run_output_heads(node_reps, span_type_reps, edge_reps, rel_type_reps)
                #prune graph pre loss calc
                if self.config.graph_prune_loc == 'pre_loss':
                    with record_function("step_15.2: prune graph pre loss calc"):
                        logits_span, logits_rel = self.prune_graph_logits(logits_span, filter_score_nodes, logits_rel, filter_score_edges)

        else:    #if we bypass the graph
            with record_function("step_15.1: run output heads"):
                logits_span, logits_rel = self.run_output_heads(cand_span_reps, span_type_reps, cand_rel_reps, rel_type_reps)

        #final processing and return
        total_loss = None
        #########################################################
        #do loss if we have labels
        #########################################################
        if has_labels:
            with record_function("step_16: calc losses"):
                #Compute losses for spans and rels final classifier heads using the span/rel type reps
                #NOTE: uses CELoss for unilabels and BCELoss for multilabels
                #def classification_loss(self, logits, labels, masks, reduction='sum', label_type='unilabel'):
                pred_span_loss = classification_loss(logits_span, 
                                                    cand_span_labels, 
                                                    cand_span_masks, 
                                                    reduction  = 'sum', 
                                                    label_type = self.config.span_labels)
                pred_rel_loss = classification_loss(logits_rel, 
                                                    cand_rel_labels, 
                                                    cand_rel_masks, 
                                                    reduction  = 'sum', 
                                                    label_type = self.config.rel_labels)
                #get the lost rel_loss
                lost_rel_loss = lost_rel_penalty * self.config.lost_rel_alpha

                #get the total loss
                total_loss = sum([filter_loss_span, filter_loss_rel, lost_rel_loss, pred_span_loss, pred_rel_loss])

                print(f' filter loss span: {filter_loss_span}')
                print(f'filter loss rel: {filter_loss_rel}')
                print(f'lost_rel_loss: {lost_rel_loss}')
                print(f'pred_span_loss: {pred_span_loss}')
                print(f'pred_rel_loss: {pred_rel_loss}')


                #add the graph structure loss if use graph
                if self.config.use_graph:
                    #Compute structure loss and total loss, not filter_loss_graph is a tensor, so we have to reduce it with sum here
                    structure_loss = (filter_loss_graph * graph_masks.float()).sum()
                    #get the total loss
                    total_loss = sum([total_loss, structure_loss])
        #########################################################

        #prune graph pre loss calc
        if self.config.use_graph and self.config.graph_prune_loc == 'post_loss':
            with record_function("step_16.2: prune graph post loss calc"):
                logits_span, logits_rel = self.prune_graph_logits(logits_span, filter_score_nodes, logits_rel, filter_score_edges)

        #########################################################
        ############################################################
        ############################################################
        #clear tensors
        #NOTE: just del the tensors you do not need for now, it is fast
        #not used for now, I think just do it outside in the train loop if at all, it uses a lot of CPU time, maybe just focus on the big tensors
        clear_tensors = False
        if clear_tensors:
            with record_function("step_17: exit clear tensors"):
                keep_tensors = ['loss', 'logits_span', 'cand_span_masks', 'cand_w_span_ids', 'cand_span_labels',
                                'logits_rel', 'cand_rel_masks', 'cand_rel_ids', 'cand_rel_labels', 'lost_rel_counts']
                if (step + 1) % self.config.clear_tensor_steps == 0:
                    clear_gpu_tensors([v for k,v in locals().items() if k not in keep_tensors])
                    clear_gpu_tensors([v for k,v in x.items() if k not in keep_tensors])
                    clear_gpu_tensors([v for k,v in result.items() if k not in keep_tensors])
        ############################################################
        ############################################################
        ############################################################

        output = dict(
            loss                 = total_loss,
            ######################################
            logits_span          = logits_span,
            cand_span_masks      = cand_span_masks,
            cand_w_span_ids      = cand_w_span_ids,    #!!!!return the word token span ids for usage in evaluation (not the cand_span_ids as that could be w or sw aligned dependent on pooling)
            cand_span_labels     = cand_span_labels,
            ######################################
            logits_rel           = logits_rel,
            cand_rel_masks       = cand_rel_masks,
            cand_rel_ids         = cand_rel_ids,
            cand_rel_labels      = cand_rel_labels,
            lost_rel_counts      = lost_rel_counts,  #how many positive rels did not get into rel_reps, i.e. not missing due to filtering or misclassification
        )

        return output


