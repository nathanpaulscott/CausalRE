import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as init

###############################################
#custom imports
from .layers_transformer_encoder_flair import TransformerEncoderFlairPrompt
from .layers_transformer_encoder_hf import TransformerEncoderHFPrompt
from .layers_other import LstmSeq2SeqEncoder, TransformerEncoderTorch, GraphEmbedder, OutputLayer
from .loss_functions import classification_loss
from .span_rep import SpanRepLayer
from .filtering import FilteringLayer
from .data_processor import RelationProcessor
from .rel_rep import RelationRepLayer




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
            rel_mode           = config.rel_mode,    #what kind of rel_rep generation algo to use 
            hidden_size        = config.hidden_size, 
            ffn_ratio          = config.ffn_ratio,
            dropout            = config.dropout,
            pooling            = config.subtoken_pooling,     #whether we are using pooling or not
        )

        # filtering layer for spans and relations
        self.span_filter_head = FilteringLayer(config.hidden_size)
        self.rel_filter_head = FilteringLayer(config.hidden_size)

        # graph embedder
        self.graph_embedder = GraphEmbedder(config.hidden_size)
        
        # transformer layer
        self.trans_layer = TransformerEncoderTorch(
            config.hidden_size,
            num_heads=config.num_heads,
            num_layers=config.num_transformer_layers
        )

        #this will replace the keep head
        self.graph_filter_head = FilteringLayer(config.hidden_size)
        
        #final output heads
        '''
        NOTE: 
        for unilabels the output dim will be num pos span/rel types + 1 for the none type
        for multilabels the output dim will be the num pos span/rel types with no none type
        '''
        self.output_head_span = OutputLayer(num_types   = config.num_span_types,
                                            hidden_size = config.hidden_size,
                                            dropout     = config.dropout,
                                            use_prompt  = config.use_prompt)
        
        self.output_head_rel = OutputLayer(num_types    = config.num_rel_types,
                                           hidden_size  = config.hidden_size,
                                           dropout      = config.dropout,
                                           use_prompt   = config.use_prompt)

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
        w2sw_map if required
                
        '''
        # Process input
        #so here they add the unqiue entity and relation classes to the bert input and then separate the reps out after bert
        #so token_reps are the embeddings that come out of the encoder transfomer (either subword embeddings for no subtoken pooling, or word embeddings with subtoken pooling), span_type_reps are the reps for the entity classes, rel_type_reps are reps for relation classes etc..
        #NOTE: temp => we return both the w2sw_map list of dicts and the sw_span_ids tensor which is used in place of the span_ids tensor for the sw token aligned case.  I am deciding which one is more efficient, I suspect the sw_span_ids
        result = self.transformer_encoder_w_prompt(x)
        #read in the values from result
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
        token_reps = self.rnn(token_reps, token_masks) 

        return dict(token_reps      = token_reps, 
                    token_masks     = token_masks,
                    span_type_reps  = span_type_reps,   #will be None for use_prompt = false
                    rel_type_reps   = rel_type_reps,    #will be None for use_prompt = false
                    sw_span_ids     = sw_span_ids,      #will be None for pooling
                    w2sw_map        = w2sw_map,         #will be None for pooling
                    cls_reps        = cls_reps)         #if using HF




    def apply_graph_keep_scores_to_logits(self, logits, filter_score, keep_thd):
        '''
        Adjusts logits based on a keep head score (filter score), effectively masking out logits
        for nodes or edges that fail a specified threshold. This method is used to refine the output
        of node/edge classification tasks by incorporating confidence scores from a graph keep head,
        which determines the relevance or importance of each node or edge in the graph.

        Args:
            logits (torch.Tensor): The original logits for nodes or edges from the output heads.
                                Shape should be (batch_size, top_k_spans/rels, num_classes),
                                where num_classes depends on the specific task.
            filter_score (torch.Tensor): Logits from a keep head that provide a score indicating
                                        the likelihood that each node/edge should be kept.
                                        Shape should be (batch_size, top_k_spans/rels).
            keep_thd (float): A threshold value used to decide whether nodes or edges are kept.
                            Nodes or edges with a keep probability (sigmoid of filter_score)
                            less than this threshold are effectively masked out.

        Returns:
            torch.Tensor: The adjusted logits, where logits for nodes/edges not meeting the keep
                        threshold are set to a very large negative value (close to zero
                        influence in subsequent softmax). The shape is the same as the input logits.
        '''
        keep_prob = torch.sigmoid(filter_score)     # (batch, top_k_spans/rels)   float (between 0 and 1)
        keep_mask = (keep_prob > keep_thd).unsqueeze(-1).float()    # (batch, top_k_spans/rels)   float (0.0 to not keep or 1.0 to keep)
        adjusted_logits = logits + (1 - keep_mask) * -1e-9
        return adjusted_logits



    def binarize_labels(self, labels):
        '''
        Make the binary int labels from labels (handles the multilabel and unilabel cases)
        labels_b will be shape (batch, num_reps) int with values of 0 or 1
        '''
        if len(labels.shape) == 3:  # Multilabel case, all zero vector is a neg case, otherwise its a pos case
            return labels.any(dim=-1).to(torch.int64)  # Aggregate multilabels to binary
        else:  # Unilabel case, any label > 0 is a pos case otherwise a neg case
            return (labels > 0).to(torch.int64)



    ##################################################################################
    ##################################################################################
    ##################################################################################
    def forward(self, x, mode='train', step=None):
        '''
        x is a batch, which is a dict, with the keys being of various types as described below:
        x['tokens']     => list of ragged lists of strings => the raw word tokenized seq data as strings
        x['seq_length'] => tensor (batch) the length of tokens for each obs
        x['span_ids']   => tensor (batch, max_seq_len_batch*max_span_width, 2) => the span_ids truncated to the max_seq_len_batch * max_span_wdith
        x['span_masks'] => tensor (batch, max_seq_len_batch*max_span_width) => 1 for spans to be used (pos cases + selected neg cases), 0 for pad, invalid and unselected neg cases  => if no labels, will be all 1 for valid/non pad spans, no neg sampling
        ###########################################
        #only present if we have labels (mode is 'train'/'pred_w_labels')
        ###########################################
        x['spans']      => list of ragged list of tuples => the positive cases for each obs  => list of empty lists if no labels
        x['relations']  => list of ragged list of tuples => the positive cases for each obs  => list of empty lists if no labels
        x['orig_map']   => list of dicts for the mapping of the orig span idx to the span_ids dim 1 idx (for pos cases only, is needed in the model for rel tensor generation later) => list of empty dicts if no labels
        x['span_labels']=> tensor (batch, max_seq_len_batch*max_span_width) int for unilabels.  (batch, max_seq_len_batch*max_span_width, num_span_types) bool for multilabels

        step will be current batch idx, i.e. we just set the total number of batch runs, say there are 1000 batches in the datset and we set pbar to 2200, then step will go from 0 to 2199, i.e. each batch will be run 2x and 200 will be 3x
        
        mode is a string with either 'train'/'pred_w_labels'/'pred_no_labels'
        '''
        #Run the transformer encoder and lstm encoder layers
        result = self.transformer_and_lstm_encoder(x)
        #read in data from results or x
        token_reps      = result['token_reps']          #(batch, max_seq_len_batch, hidden) => float, sw or w token aligned depedent pooling   
        token_masks     = result['token_masks']         #(batch, max_seq_len_batch) => bool, sw or w token aligned depedent pooling   
        w_span_ids      = x['span_ids']                 #(batch, max_seq_len_batch * max_span_width, 2) => int, w aligned span_ids
        sw_span_ids     = result['sw_span_ids']         #(batch, max_seq_len_batch * max_span_width, 2) => int, sw aligned span_ids => None if pooling
        span_masks      = x['span_masks']               #(batch, max_seq_len_batch * max_span_width) => bool
        span_labels     = x['span_labels']              #(batch, max_seq_len_batch * max_span_width) int or (batch, max_seq_len_batch * max_span_width, num_span_types) bool
        num_span_types  = self.config.num_span_types    #scalar, includes the none type (idx 0) for unilabel and only pos types for multilabel
        span_type_reps  = result['span_type_reps']      #(batch, num_span_types, hidden) => float, no mask needed
        num_rel_types   = self.config.num_rel_types     #scalar, includes the none type (idx 0) for unilabel and only pos types for multilabel
        rel_type_reps   = result['rel_type_reps']       #(batch, num_rel_types, hidden) => float, no mask needed
        cls_reps        = result['cls_reps']            #will be None for flair
        #NOTE: if use_prompt = false, the span_type_reps and rel_type_reps will be None here
        #NOTE: for prompting, if unilabels the type reps will include the none type rep, if multilabels the type reps will only be for the positive types
        
        #get some dims
        batch, max_seq_len, _ = token_reps.shape

        #generate the span reps from the token reps and the span start/end idx and outputs a tensor of shape (batch, num_spans, hidden), 
        #i.e. the span reps are grouped by start idx (remember seq_len * max_span_width == num_spans)
        #output will be shape (batch, max_seq_len_batch * max_span_width, hidden) => float
        span_reps = self.span_rep_layer(token_reps, 
                                        w_span_ids  = w_span_ids, 
                                        span_masks  = span_masks, 
                                        sw_span_ids = sw_span_ids, 
                                        cls_reps    = cls_reps,
                                        span_widths = w_span_ids[:,:,1] - w_span_ids[:,:,0])

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
        #NOTE: 
        #- we send in the span_masks here to force all masked out span scores to -inf
        #- we also force all pos cases to +inf to ensure they are first (if force_poos)
        #However, there could be caess where the number of pos cases and selected_neg cases is less than the top_k_spans,
        #So we could have a few -inf scores for masked out spans...
        #This is ok, just be aware, we also generate the cand_span_mask below to indicate this 
        #output will be shapes:  (batch, num_spans), scalar
        filter_score_span, filter_loss_span = self.span_filter_head(span_reps, 
                                                                    span_masks, 
                                                                    self.binarize_labels(span_labels), 
                                                                    force_pos_cases=force_pos,
                                                                    reduction='sum')           
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
        span_idx_to_keep = sorted_span_idx[:, :top_k_spans]    #(batch, top_k_spans)
        #do the selection
        #get tensors for span_id (w_span_ids => map span boundaries to word tokens, sw_span_ids => map span boundaries to sw tokens, span_ids = w_span_ids for pooling or sw_span_ids)
        cand_w_span_ids = w_span_ids[batch_ind, span_idx_to_keep, :]  # shape: (batch, top_k_spans, 2)
        #set cand_span_ids to either cand_w_span_ids or cand_sw_span_ids
        ############################################
        cand_span_ids = cand_w_span_ids
        if self.config.subtoken_pooling == 'none':
            cand_sw_span_ids = sw_span_ids[batch_ind, span_idx_to_keep, :]  # shape: (batch, top_k_spans, 2)
            cand_span_ids = cand_sw_span_ids
        ############################################
        #get the reps, masks and labels
        cand_span_reps = span_reps[batch_ind, span_idx_to_keep]   # shape: (batch, top_k_spans, hidden)
        cand_span_masks = span_masks[batch_ind, span_idx_to_keep]    # shape: (batch, top_k_spans)
        cand_span_labels = span_labels[batch_ind, span_idx_to_keep]  # shape: (batch, top_k_spans) unilabels or (batch, top_k_spans, num_span_types) multilabels
        #cand_span_ids are already set above

        '''
        Now process relations to find the subset to include in the initial graph
        - up to this point we only have the raw rel data => x['relations'], a list of dicts, we have no tensors
        - we need to form rel_reps, rel_labels, rel_ids, rel_masks
        NOTE: we do not use the cand prefix as this is the first time we make them....
        '''
        #get the rel labels from x['relations'] with dim 1 in same order as cand_span_id dim 1 expanded to top_k_spans**2 (this is why we need to limit top_k_spans to as low as possible)
        #NOTE: the rel_masks have the diagonal set to 0, i.e self relations masked out
        #this returns rel_labels of shape (batch, top_k_spans**2) int for unilabel or (batch, top_k_spans**2, num span types) bool for multilabel
        #as well as rel_masks, rel_ids of shape (batch, top_k_spans**2), (batch, top_k_spans**2, 2) respectively
        rel_labels, rel_masks, rel_ids = self.rel_processor.get_cand_rel_tensors(x['relations'], 
                                                                                 x['orig_map'],
                                                                                 span_idx_to_keep,
                                                                                 cand_span_ids, 
                                                                                 cand_span_masks,
                                                                                 cand_span_labels)
        '''
        Make the relation reps, has several options, based on config.rel_mode:
        1) no_context => graphER => juts concat the head and tail span reps
        2) between_context => concatenate the head and tail reps with between spans rep for context => to be coded
        3) window_context => concatenate the head and tail reps with windowed context reps, i.e. window context takes token reps in a window before and after each span and attention_pools, max_pool them => to be coded
        4+) working on more options, see code for details
        '''
        #output shape (batch, top_k_spans**2, hidden)
        rel_reps = self.rel_rep_layer(cand_span_reps, 
                                      cand_span_ids, 
                                      token_reps, 
                                      token_masks,
                                      rel_masks)

        #Compute filtering scores for relations and sort them in descending order
        ###############################################################
        #Compute filtering scores and CELoss for candidate relations from the rel_reps and the rel_labels
        #NOTE: unlike the spans, the rel_reps are already aligned with the candidate spans and so are the rel_labels (relation_classes)
        #The filter score per rel in each obs is a scaled from -inf to +inf on the conf of the rel being a positive rel (>0) or none_rel (<0)
        #the loss is an accumulated metric over all rels in all obs in the batch, so one scalar for the batch, indicating how well the model can detect that a rel is postive case (an relation) or negaitve case (none rel)
        #the binary classification head is trainable, so hopefully it gets better at determining if a rel is positive over time
        #NOTE: the structure of the binary classification head and the score and loss calc is identical to the span case
        ########################################################
        #determine the postive case forcing strategy first, typically rel_fpos_forcing is not enabled
        force_pos = self.config.rel_force_pos
        if force_pos == True and (self.config.pos_force_step_limit != 'none' and self.config.pos_force_step_limit > step+1):
            force_pos = False
        #run the filter
        #output will be shapes:  (batch, top_k_spans**2), scalar
        filter_score_rel, filter_loss_rel = self.rel_filter_head(rel_reps, 
                                                                 rel_masks,
                                                                 self.binarize_labels(rel_labels),
                                                                 force_pos_cases=force_pos,
                                                                 reduction='sum')           

        # Sort the filter scores for rels in descending order and just get the rel_idx 
        #NOTE [1] is to get the idx as opposed to the values as torch.sort returns a tuple
        sorted_rel_idx = torch.sort(filter_score_rel, dim=-1, descending=True)[1]    #(batch, top_k_spans**2)
        #do the relation pruning
        #Calculate a dynamic top_k for relations
        #returns a scalar which should be << top_k_spans**2
        top_k_rels = self.rel_processor.dynamic_top_k_rels(filter_score_rel, self.config)
        #Select the top_k relationships for processing
        batch_indices = torch.arange(batch).unsqueeze(-1)   #(batch, 1)
        rel_idx_to_keep = sorted_rel_idx[:, :top_k_rels]   #(batch, top_k_rels)
        #Extract the candidate relationship representations and labels
        cand_rel_reps   = rel_reps[batch_indices, rel_idx_to_keep]    #(batch, top_k_rels, hidden)  float
        cand_rel_masks  = rel_masks[batch_indices, rel_idx_to_keep]      #(batch, top_k_rels)  bool
        cand_rel_labels = rel_labels[batch_indices, rel_idx_to_keep]     #(batch, top_k_rels) int for unilabel or (batch, top_k_rels, num rel types) int for multilabel 
        cand_rel_ids    = rel_ids[batch_indices, rel_idx_to_keep]     #(batch, top_k_rels, 2) int

        #Generate the node and edge reps
        #this basically superimposes a node identifier to the span_reps and an edge identifier to the rel_reps
        #i.e. the id is added by element wise addition to the reps, the same way as pos encoding is added
        #this is preparing the reps for graph attention
        #node reps with be shape (batch, top_k_spans, hidden)
        #edge reps with be shape (batch, tp_k_rels, hidden)
        node_reps, edge_reps = self.graph_embedder(cand_span_reps, cand_rel_reps, cand_span_masks, cand_rel_masks)

        #merge the node and edge reps into one tensor so we can run it through a attention block
        #think of it like a sequence and each span or pair is a token, so he is concat the spans and pairs
        #so shape will be (batch, top_k_spans + top_k_rels, hidden)
        graph_reps = torch.cat((node_reps, edge_reps), dim=1)   #shape (batch, top_k_spans + top_k_rels, hidden)   float
        #merge masks
        graph_masks = torch.cat((cand_span_masks, cand_rel_masks), dim=1)   #(batch, top_k_spans + top_k_rels)   bool
        #binarize and merge labels as this is just used for graph pruning
        graph_labels_b = torch.cat((self.binarize_labels(cand_span_labels), 
                                    self.binarize_labels(cand_rel_labels)), dim=1)   #(batch, top_k_spans + top_k_rels) int (0,1) for unilabel/multilabel as they have been binarized

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
        graph_reps = self.trans_layer(graph_reps, graph_masks)
        
        #run graph reps through a binary classification head
        #determine the postive case forcing strategy first, typically rel_pos_forcing is not enabled
        force_pos = self.config.graph_force_pos
        if force_pos == True and (self.config.pos_force_step_limit != 'none' and self.config.pos_force_step_limit > step+1):
            force_pos = False
        #run the graph filter (the keep filter)
        #output will be shapes:  (batch, top_k_spans + top_k_rels), scalar
        filter_score_graph, filter_loss_graph = self.graph_filter_head(graph_reps, 
                                                                       graph_masks,
                                                                       graph_labels_b,
                                                                       force_pos_cases=force_pos,
                                                                       reduction='none')           

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
        rels
        - edge_reps, 
        - filter_score_edges,
        - cand_rel_masks, 
        - cand_rel_labels, 
        - cand_rel_ids
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
        #####################################################################################
        '''
        if self.config.use_prompt:
            logits_span = self.output_head_span(node_reps, span_type_reps)  # Shape: (batch, top_k_spans, num_span_types)
            logits_rel = self.output_head_rel(edge_reps, rel_type_reps)   # Shape: (batch, top_k_rels, num_rel_types)
        else:
            logits_span = self.output_head_span(node_reps)  # Shape: (batch, top_k_spans, num_span_types)
            logits_rel = self.output_head_rel(edge_reps)   # Shape: (batch, top_k_rels, num_rel_types)

        #prune graph pre loss calc
        '''
        NOTE: the pruning is done by biasing the logits to very negative (-1e9) for nodes/edges that we want to prune out
              this effectively masks out these nodes/edges from all downstream loss/prediction functions
        '''
        if self.config.graph_prune_loc == 'pre_loss':
            logits_span = self.apply_graph_keep_scores_to_logits(logits_span, filter_score_nodes, self.config.node_keep_threshold)
            logits_rel = self.apply_graph_keep_scores_to_logits(logits_rel, filter_score_edges, self.config.edge_keep_threshold)

        #final processing and return
        total_loss = None
        if mode in ['train', 'pred_w_labels']:
            #Compute losses for spans and rels final classifier heads using the span/rel type reps
            #NOTE: uses CELoss for unilabels and BCELoss for multilabels
            pred_span_loss = classification_loss(logits_span, 
                                                 cand_span_labels, 
                                                 cand_span_masks, 
                                                 reduction='sum', 
                                                 label_type=self.config.span_labels)
            pred_rel_loss = classification_loss(logits_rel, 
                                                cand_rel_labels, 
                                                cand_rel_masks, 
                                                reduction='sum', 
                                                label_type=self.config.rel_labels)
            #Compute structure loss and total loss, not filter_loss_graph is a tensor, so we have to reduce it with sum here
            structure_loss = (filter_loss_graph * graph_masks.float()).sum()
            #get the total loss
            total_loss = sum([filter_loss_span, filter_loss_rel, pred_span_loss, pred_rel_loss, structure_loss])

        #prune graph pre loss calc
        '''
        NOTE: the pruning is done by biasing the logits to very negative (-1e9) for nodes/edges that we want to prune out
              this effectively masks out these nodes/edges from all downstream prediction functions
        '''
        if self.config.graph_prune_loc == 'post_loss':
            logits_span = self.apply_graph_keep_scores_to_logits(logits_span, filter_score_nodes, self.config.node_keep_threshold)
            logits_rel = self.apply_graph_keep_scores_to_logits(logits_rel, filter_score_edges, self.config.edge_keep_threshold)

        output = dict(
            loss                 = total_loss,
            ######################################
            logits_span          = logits_span,
            cand_span_masks      = cand_span_masks,
            cand_span_ids        = cand_span_ids,
            cand_span_labels     = cand_span_labels,
            top_k_spans          = top_k_spans,
            ######################################
            logits_rel           = logits_rel,
            cand_rel_masks       = cand_rel_masks,
            cand_rel_ids         = cand_rel_ids,
            cand_rel_labels      = cand_rel_labels,
            top_k_rels           = top_k_rels
        )

        return output





'''
Got to here
Got to here
Got to here
Got to here
Got to here
ok, so I am going thorugh this matching loss business, I see somehting weird going on
I think it is incorrect for now, I think you only need 1 function, I have 2 for promtp and no prompt
go through it in detail and work out what the fuck is going on, I know hte output heads are different for the promtp/no prompt case, but I think the shapes are hte same
Just check it all....

##################################################
##################################################
##################################################
##################################################
The current maytching loss is incorrect.  Check what format the span and rel labels are in , are they onelabel per span/rel or a binary vector.  If one value then use CELoss, if a binary vetcor, then use BCELoss.
START here, this needs to be sorted now to stop confusion, it affects the loss and alot of filtering loss calcs throughout the model!!!!
##################################################
##################################################
##################################################



Then look at the preds and how we get them, shpould they still be tensors, shoud they go to python objects, I know at some poitn we need human readable output, but for the metrics, can we not stay in tensors?????

Then look at how the metrics are calcd in the train/pred_w_labels case, they should be calcd on the preds/labels, so on the tensors or the python objects, would be easier if on the tensors


look at the matching loss, why is he using binary cross entropy loss for both spans and relations?.  Spans would use CELoss (only one possible positive class per span) and rels would use BCELoss (multiple possible positive cases per rel)
You coudl even look at using CELoss for rels also.

I guess we can use BCELoss for both spans and rels, have a look at it and if it makes sense.

Additional thoughts.
If we get issues woth the long spans and trying to get positive predictions for these complex events/states, we will have to break it up into event/state triggers (spans) and event/state args (rels) to identify the complete event/state spans.
This would essentially use the same structure as we have now, except that labels will be different, you would need instead of spans and rels, you would need: trigger_args_spans (start, end, type), arg_trigger_rels (arg_head, trigger_tail, type), trigger_trigger_rels (trigger_head, trigger_tail, type)
Where the trigger_trigger rels are the inter event/state trigger, each event/state has one trigger and zero or more args.  
So the flow would be:
1) trigger and arg span classification, the benefit of this is the trigger span length is much less than what I was planning, say 10-12 word tokens
2) arg to trigger rel classification, this is completing the event/states, the output of this would be the ability to generate the event/state reps, which could be somehting like trigger_rep + args_rep, where args_rep is the combined rep for all args of a trigger, maybe combine with attention pooling with the trigger as the query
3) trigger to trigger rel classification, form the trigger to trigger rels, in various ways, i.e. concat event_reps, potentially add context.  Then classifiy
Thus the main difference to the current flow is that we have 2 rel stages, which would not be that hard to add in to tthe current structure, I would follow a similar approach.
1) get the trigger and arg spans => prune the triggers and args with a filter head as we do now, the make the trigger and arg reps.  For this stage ignore any dependency of the trigger and args, that will come out in stage 2
2) then make the arg to trigger reps and run through a filter head and prune
3) classify the arg to trigger rels through an output head => this will give you the initial event/state graph and make the event/state reps
4) classify the event to event rels => make the event relation reps pass through filtering head and prune.
5) make the graph nodes are event reps and rels are event to event reps
6) pass through a graph transformer
7) split back to nodes edge reps and pass each through an output layer
8) filter by the graph pruning filter scores
9) run the loss and output


'''