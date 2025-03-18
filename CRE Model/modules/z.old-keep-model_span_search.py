import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as init
import torch.utils.checkpoint as checkpoint

from torch.profiler import record_function

from types import SimpleNamespace
import copy


###############################################
#custom imports
from .layers_transformer_encoder_hf import TransformerEncoderHF
from .layers_filtering import FilteringLayerBinaryDouble, FilteringLayerBinarySingle
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
        #set the num_limit
        self.num_limit = 6.5e4 if self.config.num_precision == 'half' else 1e9
        
        #make the modified transformer encoder with subtoken pooling functionality
        self.transformer_encoder = TransformerEncoderHF(self.config)
        
        #bilstem layer to mix the embeddings around even more
        if self.config.use_lstm:
            self.rnn = LstmSeq2SeqEncoder(
                input_size      = self.config.hidden_size,
                hidden_size     = self.config.hidden_size // 2,
                num_layers      = self.config.lstm_layers,
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
            cls_flag              = self.config.backbone_model_source == 'HF'    #whether we will have a cls token rep
        )

        #set the FilteringLayer
        FilteringLayer = FilteringLayerBinaryDouble
        if self.config.filter_head_type == 'single':
            FilteringLayer = FilteringLayerBinarySingle

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
        GraphFilteringLayer = FilteringLayerBinaryDouble
        if self.config.graph_filter_head_type == 'single':
            GraphFilteringLayer = FilteringLayerBinarySingle
        
        self.graph_filter_head = GraphFilteringLayer(self.config.hidden_size, 
                                                     num_limit = self.num_limit,
                                                     dropout   = self.config.dropout if self.config.filter_dropout else None) 
        
        #final output heads
        '''
        NOTE: 
        for unilabels the output dim will be num pos span/rel types + 1 for the none type
        for multilabels the output dim will be the num pos span/rel types with no none type
        NOTE: why the unilabel case needs the none type => because it uses cross entropy, if you do not give it the option to pred a none-type
        then it must choose one of the pos type even if none are any good, you will get all pos cases => you must give it the option to have neg cases
        '''
        self.output_head_span = OutputLayer(input_size  = self.config.hidden_size,
                                            output_size = self.config.num_span_types,
                                            dropout     = self.config.dropout)
        
        self.output_head_rel = OutputLayer(input_size   = self.config.hidden_size,
                                           output_size  = self.config.num_rel_types,
                                           dropout      = self.config.dropout)

        self.init_weights()



    def init_weights(self):
        #you need to init weights here
        pass


    def transformer_and_lstm_encoder(self, x):
        '''
        This just gets the device internally

        Inputs: x
        
        Returns:
        token_reps/masks
        span_reps/masks
        sw_span_ids
        '''
        #Process input
        #token_reps are the embeddings that come out of the encoder transfomer (either subword embeddings for no subtoken pooling, or word embeddings with subtoken pooling)
        result = self.transformer_encoder(x)
        #read in the values from result
        token_reps     = result['token_reps']       #embeddings of the word/sw tokens
        token_masks    = result['token_masks']      #masks for the word/sw tokens
        cls_reps       = result['cls_reps']         #embeddings for the CLS sw token, only if we are using HF
        sw_span_ids    = result['sw_span_ids']      #tensor (batch, batch_max_seq_len*max_span_width, 2) => x['span_ids'] with values mapped using w2sw_map to the sw token start, end. Only if we are HF with no pooling.

        #Enrich token_reps
        #enriches the (batch, seq_len, hidden) shaped token reps with a bilstm to (batch, seq_len, hidden) shaped token reps, 
        #not clear if this is really necessary, but may be to deal with the side effects of first sw token pooling to get word tokens, 
        #at least that is what they allude to in their associated paper https://arxiv.org/pdf/2203.14710
        if self.config.use_lstm:
            token_reps = self.rnn(token_reps, token_masks) 

        return dict(token_reps      = token_reps, 
                    token_masks     = token_masks,
                    sw_span_ids     = sw_span_ids,      #will be None for pooling
                    cls_reps        = cls_reps)         #if using HF



    def calc_top_k_spans(self, span_scores):
        '''
        Calculates the maximum number of significant spans to include for this batch based on a quantile threshold. 
        It determines the number of spans whose scores exceed the calculated threshold and limits this number based on configuration constraints and the maximum possible spans that can be included.

        Args:
            span_scores (torch.Tensor): Tensor containing scores of spans, with higher scores indicating a stronger likelihood of significance.

        Returns:
            int: The number of spans to include, determined as the minimum of the highest count of spans exceeding the threshold across the batch, the maximum number allowed by configuration, and the total number of possible spans.
        '''
        #Calc the max number of desired spans based on a quantile calculation
        if self.config.num_precision == 'half':
            # Quantile doesn't support half precision, convert dtype to float32
            span_score_threshold = torch.quantile(span_scores.to(dtype=torch.float32), self.config.span_score_percentile / 100)
        else:
            span_score_threshold = torch.quantile(span_scores, self.config.span_score_percentile / 100)
        desired_spans_count = (span_scores >= span_score_threshold).sum(dim=1) 
        batch_max_desired_spans = desired_spans_count.max().item()  # Max across the batch to ensure we consider the highest valid count
        
        #Calculate the maximum number of spans that can be included based on validity
        S = self.batch_max_seq_len
        W_mod = min(S, self.config.max_span_width)
        num_spans_available = S * W_mod - W_mod * (W_mod - 1) // 2

        # Determine the final number of top_k_spans
        top_k_spans = min(batch_max_desired_spans, self.config.max_top_k_spans, num_spans_available)

        return int(top_k_spans)



    def calc_top_k_rels(self, rel_scores):
        """
        Calculates the maximum number of significant relationships to include for this batch based on a quantile threshold. 
        It determines the number of relationships whose scores exceed the calculated threshold and limits this number based on configuration constraints and the actual number of relationships available.

        Args:
            rel_scores (torch.Tensor): Tensor containing scores of relationships, with higher scores indicating a stronger likelihood of the relationship being significant.

        Returns:
            int: The number of relationships to include, determined as the minimum of the highest count of relationships exceeding the threshold across the batch, the maximum number allowed by configuration, and the total number of relationships available.
        """
        #determine thebatch max number of desired rels
        if self.config.num_precision == 'half':
            # Quantile doesn't support half precision, convert dtype to float32
            rel_score_thd = torch.quantile(rel_scores.to(dtype=torch.float32), self.config.rel_score_percentile / 100)
        else:
            rel_score_thd = torch.quantile(rel_scores, self.config.rel_score_percentile / 100)
        desired_rels_count = (rel_scores >= rel_score_thd).sum(dim=1)  # Calculate valid relations for each item in the batch
        batch_max_desired_rels = desired_rels_count.max().item()  # Get the maximum valid relations count across the batch

        #determine the max number of available rels
        num_rels_available = rel_scores.shape[1]
        
        top_k_rels = min(batch_max_desired_rels, self.config.max_top_k_rels, num_rels_available)

        return int(top_k_rels)



    def prune_spans(self, filter_score_span, top_k_spans):
        '''
        select the top K spans for the initial graph based on the span filter scores
        This is our smart neg sampling for spans
        NOTE: it is possible in some cases with short sequences, that some of the shortlisted top_k_spans 
        are masked out, this is ok, it is captured in the cand_span_masks, I'm just highlighting it
        For most cases however, the number of available spans far outnumber the max_top_k_spans
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
        This is our smart neg sampling for rels
        NOTE: it is possible in some cases with short sequences, that some of the shortlisted top_k_rels 
        are masked out, this is ok, it is captured in the cand_rel_masks, I'm just highlighting it
        For most cases however, the number of available rels far outnumber the max_top_k_rels
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
        labels_b will be shape (batch, num_reps) bool with values of 0 or 1
        NOTE: returns None if labels is None
        '''
        if labels is None:
            return None
        
        if len(labels.shape) == 3:  # Multilabel case, all zero vector is a neg case, otherwise its a pos case
            return labels.any(dim=-1)
        else:  # Unilabel case, any label > 0 is a pos case otherwise a neg case
            return (labels > 0)


    def set_force_pos(self, type, step):
        '''
        This determines if force pos cases is True/False dependent on the configs and the current step and the filter type
        '''
        #No teacher forcing for non training
        if not self.training:
            return False

        if type == 'span':
            if self.config.filter_force_pos == 'temp':
                if step+1 > self.config.force_pos_step_limit:
                    return False
                else:
                    return True
            return True

        if type == 'rel':
            if self.config.rel_force_pos == 'never':
                return False
            else:    #if it is 'follow'
                if self.config.filter_force_pos == 'temp':
                    if step+1 > self.config.force_pos_step_limit:
                        return False
                    else:
                        return True
                return True

        return False
 

    def calc_pred_losses(self, logits_span, cand_span_labels, cand_span_masks, 
                               logits_rel, cand_rel_labels, cand_rel_masks):
        #span pred loss
        pred_span_loss = classification_loss(logits_span, 
                                             cand_span_labels, 
                                             cand_span_masks, 
                                             reduction     = self.config.loss_reduction, 
                                             label_type    = 'unilabel')

        #rel pred loss
        pred_rel_loss = classification_loss(logits_rel, 
                                            cand_rel_labels, 
                                            cand_rel_masks, 
                                            reduction     = self.config.loss_reduction, 
                                            label_type    = self.config.rel_labels)

        return pred_span_loss, pred_rel_loss
    



    def fixed_span_filter(self, span_binary_labels, span_masks):
        '''
        A temporary hack to bypass the span filter head with random neg sampling just for testing, leave in as would be useful to show later.
        Negative sampling is limited to indices where span_masks is True.
        It will output filter_scores of pos_limit for all pos cases and 0 for selected neg cases (who are not masked out) and neg limit for unselected neg cases and masked out spans
        '''
        # Set positive and negative limits
        pos_limit = self.num_limit
        neg_limit = -self.num_limit
        
        # Initialize span_filter_score with negative limits
        filter_score_span = torch.full_like(span_masks, neg_limit, dtype=torch.float32)
        # Set scores for positive labels
        filter_score_span[span_binary_labels] = pos_limit
        
        # Determine the number of negative samples to select as zero based on configuration
        num_pos_cases = torch.sum(span_binary_labels, dim=1)  # Sum along the span axis for each batch
        max_neg_samples = self.config.max_top_k_spans - num_pos_cases
        
        # For each batch, randomly select negatives to set to zero
        for i in range(filter_score_span.shape[0]):  # iterate over the batch dimension
            # Find indices of negative cases where the mask is True
            neg_indices = ((span_binary_labels[i] == False) & (span_masks[i] == True)).nonzero(as_tuple=False)
            # Randomly select negatives to set to zero, limited by max_neg_samples[i]
            if neg_indices.numel() > 0:  # Check if there are any eligible negative indices
                num_neg_samples = min(max_neg_samples[i].item(), neg_indices.shape[0])
                selected_neg_indices = neg_indices[torch.randperm(neg_indices.shape[0])[:num_neg_samples]]
                filter_score_span[i, selected_neg_indices] = 0

        return filter_score_span, None


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
        span_masks      = x['span_masks']               #(batch, batch_max_seq_len * max_span_width) => bool, includes the spans that are pos cases and selected neg cases for each batch obs
        span_labels     = x['span_labels']              #(batch, batch_max_seq_len * max_span_width) int
        num_span_types  = self.config.num_span_types    #scalar, includes the none type (idx 0) for unilabel and only pos types for multilabel
        num_rel_types   = self.config.num_rel_types     #scalar, includes the none type (idx 0) for unilabel and only pos types for multilabel
        cls_reps        = result['cls_reps']            #will be None for flair
        
        #######################################################
        # SPANS ###############################################
        #######################################################
        '''
        #NOTE: if you wanted to chunk the span filtering to handle long seq, long spans cases, you would do it here on teh whole make span reps and run the span filter head chain
        #just pass in a modified span mask for each chunk and mod the code in teh span rep extractor to only extract reps for actuive chunk spans, not all possible spans with unused ones set to 0!!
        #you just have to ensure the shpaes all line up, as right now it assumes num_spans for dim 1, so if you are going to only consider active spans in each chunk then you have to speep all the code so they all align
        #span_widths, span_masks, sw_span_ids, w_span_ids and internal code, check all, not so hard, but check all
        #then the output for each chunk will be (batch, num_chunk_spans, hidden) whcih goes to the filter head, then we take the filter scores for each chunk andmarge them and sort them, along wiht a mapping of the chunk it came from and the idx in that chunk
        kind of fiddly, then we have to select those reps from each chunk and stack to a final cand_span_reps, masks, labels etc..  
        I would remove all extraneous stuff first, like sw tokens, pos weight etc..  Just remove it
        The other change would be to checkpoint the binary filter head so the backwards pass doesn't die
        
        '''
        #generate the span reps from the token reps and the span start/end idx and outputs a tensor of shape (batch, num_spans, hidden), 
        #i.e. the span reps are grouped by start idx
        #NOTE: for all further analyses => num_spans is the total num spans including those that end outside the max seq len, they are just masked out
        #span_reps shape (batch, batch_max_seq_len * max_span_width, hidden) => float
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
        #However, there could be cases where the number of pos cases and selected_neg cases is less than the top_k_spans,
        #So we could have a few -inf scores for masked out spans...
        #This is ok, just be aware, we also generate the cand_span_mask below to indicate this 
        #output will be shapes:  (batch, num_spans), scalar 
        #loss will be 0 for has_labels = False
    
        #temp for testing
        bypass_span_filter = False
        if bypass_span_filter:   #this is a random neg sampling for testing
            filter_score_span, filter_loss_span = self.fixed_span_filter(self.binarize_labels(span_labels), span_masks)
        else:
            force_pos_cases = self.set_force_pos('span', step)
            filter_score_span, filter_loss_span = self.span_filter_head(span_reps, 
                                                                        span_masks, 
                                                                        self.binarize_labels(span_labels) if has_labels else None, 
                                                                        force_pos_cases = force_pos_cases,
                                                                        reduction       = self.config.loss_reduction,
                                                                        loss_only       = False)           

        #get top_k_spans
        top_k_spans = self.calc_top_k_spans(filter_score_span)

        #prune the spans to top_k_spans
        span_idx_to_keep, span_to_cand_span_map = self.prune_spans(filter_score_span, top_k_spans) 
        #filter the reps, masks and labels
        cand_span_reps = span_reps[self.batch_ids, span_idx_to_keep]   # shape: (batch, top_k_spans, hidden)
        if bypass_span_filter:   #this is a random neg sampling for testing
            updated_span_masks = span_masks & (filter_score_span > -self.num_limit)
            cand_span_masks = updated_span_masks[self.batch_ids, span_idx_to_keep]    # shape: (batch, top_k_spans)
        else:
            cand_span_masks = span_masks[self.batch_ids, span_idx_to_keep]    # shape: (batch, top_k_spans)
        cand_span_labels = span_labels[self.batch_ids, span_idx_to_keep] if has_labels else None  # shape: (batch, top_k_spans) unilabels or (batch, top_k_spans, num_span_types) multilabels
        cand_w_span_ids = w_span_ids[self.batch_ids, span_idx_to_keep, :]  # shape: (batch, top_k_spans, 2)
        #set cand_span_ids to either cand_w_span_ids or cand_sw_span_ids
        cand_span_ids = cand_w_span_ids
        if self.config.subtoken_pooling == 'none':
            cand_sw_span_ids = sw_span_ids[self.batch_ids, span_idx_to_keep, :]  # shape: (batch, top_k_spans, 2)
            cand_span_ids = cand_sw_span_ids


        ###############################################################
        # RELATIONS ###################################################
        ###############################################################
        #get the rel labels from x['relations'] with dim 1 in same order as cand_span_id dim 1 expanded to top_k_spans**2 (this is why we need to limit top_k_spans to as low as possible)
        #NOTE: the rel_masks have the diagonal set to 0, i.e self relations masked out
        #this returns rel_labels of shape (batch, top_k_spans**2) int for unilabel or (batch, top_k_spans**2, num span types) bool for multilabel => None if not has_labels
        #as well as rel_masks, rel_ids of shape (batch, top_k_spans**2), (batch, top_k_spans**2, 2) respectively
        #if has_labels, it also returns the lost_rel_cnts tensor of shape (batch) which is one int per batch obs indicating the number fo rels that had annotations but did not make it to get rel_ids as they were filtered out byt the span filtering
        #we will add a lost rel penalty loss for this
        rel_ids, rel_masks, rel_labels, lost_rel_counts, lost_rel_penalty = self.rel_processor.get_cand_rel_tensors(cand_span_masks,       #the span mask for each selected span  (batch, top_k_spans)
                                                                                                                    x['relations'],        #the raw relations data.  NOTE: will be None for no labels
                                                                                                                    x['orig_map'],         #maps raw span idx to span_ids idx.  NOTE: will be None for no labels  
                                                                                                                    span_to_cand_span_map) 

        '''
        Make the relation reps, has several options, based on config.rel_mode:
        1) no_context => graphER => juts concat the head and tail span reps
        2) between_context => concatenate the head and tail reps with between spans rep for context => to be coded
        3) window_context => concatenate the head and tail reps with windowed context reps, i.e. window context takes token reps in a window before and after each span and attention_pools, max_pool them => to be coded
        4+) working on more options, see code for details
        '''
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
        force_pos_cases = self.set_force_pos('rel', step)
        filter_score_rel, filter_loss_rel = self.rel_filter_head(rel_reps, 
                                                                 rel_masks, 
                                                                 self.binarize_labels(rel_labels) if has_labels else None, 
                                                                 force_pos_cases = force_pos_cases,
                                                                 reduction       = self.config.loss_reduction,
                                                                 loss_only       = False)           

        #Calculate the number of rels to shortlist
        top_k_rels = self.calc_top_k_rels(filter_score_rel)   #returns a scalar which should be << top_k_spans**2
        #prune the rels
        rel_idx_to_keep = self.prune_rels(filter_score_rel, top_k_rels)
        cand_rel_reps   = rel_reps[self.batch_ids, rel_idx_to_keep]    #(batch, top_k_rels, hidden)  float
        cand_rel_masks  = rel_masks[self.batch_ids, rel_idx_to_keep]      #(batch, top_k_rels)  bool
        cand_rel_ids    = rel_ids[self.batch_ids, rel_idx_to_keep]     #(batch, top_k_rels, 2) int
        cand_rel_labels = rel_labels[self.batch_ids, rel_idx_to_keep] if has_labels else None     #(batch, top_k_rels) int for unilabel or (batch, top_k_rels, num rel types) int for multilabel 
        

        #########################################################
        #the graph transformer section
        #########################################################
        if not self.config.use_graph:    #if we bypass the graph completely
            node_reps, edge_reps = cand_span_reps, cand_rel_reps
        
        else:    #use the graph transformer
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
            binary_graph_labels = torch.cat((self.binarize_labels(cand_span_labels), self.binarize_labels(cand_rel_labels)), dim=1) if has_labels else None   #(batch, top_k_spans + top_k_rels) int (0,1) for unilabel/multilabel as they have been binarized

            ###################################################
            #Apply transformer layer and keep_head
            #Q: why not use a GAT => beacuse the GAT doesn't enrich the edge reps, only the node reps, even if we get it to take in the edge reps, it is garbage
            ###################################################
            #this is a using the torch built in mha transformer encoder, mask is the key padding mask
            #shape out will be same as input reps (batch, top_k_spans + top_k_rels, hidden)
            graph_reps = self.trans_layer(graph_reps, graph_masks)
            #run graph reps through a binary classification head
            #output will be shapes:  None, scalar (None if has_labels is False)
            _, filter_loss_graph = self.graph_filter_head(graph_reps, 
                                                          graph_masks,
                                                          binary_graph_labels,    #will be None if has_labels is False
                                                          force_pos_cases = False,   #should never be True as filter scores are not used for the Graph filter, only the loss 
                                                          reduction       = self.config.loss_reduction,
                                                          loss_only       = True)           

            #Split the reps back to nodes(spans) and edges(rels)
            node_reps, edge_reps = graph_reps.split([top_k_spans, top_k_rels], dim=1)                             #(batch, top_k_spans + top_k_rels, hidden) => (batch, top_k_spans, hidden), (batch, top_k_rels, hidden)
            
            #set the node and edge reps back to the pre-graph transformer cand_span_reps and cand_rel_reps if use_graph_reps is False
            #in this case only the graph loss is used from teh graph processing
            if not self.config.use_graph_reps:
                node_reps, edge_reps = cand_span_reps, cand_rel_reps



        ############################################################################
        #Output Heads
        ############################################################################
        ############################################################################
        #NOTE: num_span_types and num_rel_types are the number of neg/pos types for unilabel and pos types only for multilabel
        logits_span = self.output_head_span(node_reps)  # Shape: (batch, top_k_spans, num_span_types)
        logits_rel = self.output_head_rel(edge_reps)   # Shape: (batch, top_k_rels, num_rel_types)

        #final processing and return
        total_loss = None
        #########################################################
        #do loss if we have labels
        #########################################################
        if has_labels:
            #Compute losses for spans and rels final classifier heads using the span/rel type reps
            #NOTE: uses CELoss for unilabels and BCELoss for multilabels
            pred_loss_span, pred_loss_rel = self.calc_pred_losses(logits_span,
                                                                  cand_span_labels, 
                                                                  cand_span_masks, 
                                                                  logits_rel,
                                                                  cand_rel_labels, 
                                                                  cand_rel_masks)

            #get the lost rel_loss (only will have values if we are using non teacher forcing for spans, i.e. temp and past the warmup)
            lost_rel_loss = lost_rel_penalty * self.config.lost_rel_alpha
            
            #get the total loss
            if self.config.use_graph:
                total_loss = sum([filter_loss_span, filter_loss_rel, lost_rel_loss, filter_loss_graph, pred_loss_span, pred_loss_rel])
                #total_loss = sum([lost_rel_loss, filter_loss_graph, pred_loss_span, pred_loss_rel])
                #total_loss = sum([lost_rel_loss, pred_loss_span, pred_loss_rel])
            else:
                total_loss = sum([filter_loss_span, filter_loss_rel, lost_rel_loss, pred_loss_span, pred_loss_rel])
                #total_loss = sum([lost_rel_loss, pred_loss_span, pred_loss_rel])
                #total_loss = pred_loss_span

        #print(f' fls: {filter_loss_span}, flr: {filter_loss_rel}, lrl: {lost_rel_loss}, flg: {filter_loss_graph if self.config.use_graph else "-"}, cls: {pred_loss_span}, clr: {pred_loss_rel}')

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

        #print(f'top_k_spans: {top_k_spans}, top_k_rels: {top_k_rels}')

        return output
