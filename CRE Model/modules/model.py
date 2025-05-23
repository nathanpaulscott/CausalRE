import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as init
import torch.utils.checkpoint as checkpoint
from torch.nn.utils.rnn import pad_sequence

from torch.profiler import record_function

from types import SimpleNamespace
import copy


###############################################
#custom imports
from .layers_transformer_encoder_hf import TransformerEncoderHF
from .layers_token_tagging import TokenTagger
from .layers_filtering import FilteringLayerBinaryDouble, FilteringLayerBinarySingle
from .layers_other import LstmSeq2SeqEncoder, TransformerEncoderTorch, GraphEmbedder, GraphTransformerModel, OutputLayer
from .loss_functions import classification_loss
from .utils import clear_gpu_tensors

from .span_rep import SpanRepLayer

from .rel_processor import RelationProcessor

from .rel_rep import RelationRepLayer



class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        #create a locally mutable version of the config namespace
        self.config = SimpleNamespace(**config.get_data_copy())

        #make the modified transformer encoder with subtoken pooling functionality
        if self.config.bert_shared_unmarked_span_rel:
            self.transformer_encoder_span = TransformerEncoderHF(self.config)
            self.transformer_encoder_rel = self.transformer_encoder_span
        else:
            self.transformer_encoder_span = TransformerEncoderHF(self.config)
            self.transformer_encoder_rel = TransformerEncoderHF(self.config)
        
        #bilstm layer to mix the embeddings around even more
        if self.config.use_lstm:
            self.rnn = LstmSeq2SeqEncoder(input_size      = self.config.hidden_size,
                                          hidden_size     = self.config.hidden_size // 2,
                                          num_layers      = self.config.lstm_layers,
                                          bidirectional   = True,
                                          skip            = self.config.lstm_skip_con)

        #define the token tagger
        self.token_tag_layer = TokenTagger(tagging_mode   = self.config.tagging_mode,
                                           #rest are in kwargs
                                           input_size     = self.config.hidden_size,
                                           num_limit      = self.config.num_limit, 
                                           max_span_width = self.config.max_span_width,
                                           dropout        = self.config.dropout,
                                           predict_thd    = self.config.predict_thd)

        #span width embeddings (in word widths)
        #NOTE: the number of embeddings needs to be max_span_width + 1 as the first one (idx 0) can be used for widths of length 0 (which are to be ignored, i.e. these are masked out spans anyway)
        #NOTE: do not need to specify the float dtype as we are using autocast
        self.width_embeddings = nn.Embedding(self.config.max_span_width + 1, 
                                             self.config.width_embedding_size) 
        #span representations
        #this forms the span reps from the token reps using the method defined by config.span_mode,
        self.span_rep_layer = SpanRepLayer(
            span_mode             = self.config.span_mode,
            max_span_width        = self.config.max_span_width,    #in word widths
            #rest are in kwargs
            hidden_size           = self.config.hidden_size,
            layer_type            = self.config.projection_layer_type,
            ffn_ratio             = self.config.ffn_ratio, 
            width_embeddings      = self.width_embeddings if self.config.use_width_embedding else None,    #in word widths
            dropout               = self.config.dropout,
            cls_flag              = self.config.use_cls_embedding and self.config.backbone_model_source == 'HF'    #whether we will have a cls token rep
        )


        self.span_rep_layer_for_rels = SpanRepLayer(
            span_mode                = self.config.span_mode,
            max_span_width            = self.config.max_span_width,    #in word widths
            #rest are in kwargs
            hidden_size               = self.config.hidden_size,
            layer_type                = self.config.projection_layer_type,
            ffn_ratio                 = self.config.ffn_ratio, 
            width_embeddings          = None,     #self.width_embeddings if self.config.use_width_embedding else None,    #in word widths
            dropout                   = self.config.dropout,
            cls_flag                  = False
        )


        #set the FilteringLayer
        FilteringLayer = FilteringLayerBinaryDouble
        if self.config.filter_head_type == 'single':
            FilteringLayer = FilteringLayerBinarySingle

        #filtering layer for spans
        self.span_filter_head = FilteringLayer(self.config.hidden_size, 
                                               num_limit = self.config.num_limit,
                                               dropout   = self.config.dropout if self.config.filter_dropout else None) 

        #define the relation processor to process the raw x['relations'] data once we have our initial span_ids
        self.rel_processor = RelationProcessor(self.config)
        
        #relation representation
        #this forms the rel reps from the span_reps after the span reps have been filtered for the initial graph
        self.rel_rep_layer = RelationRepLayer(
            rel_mode           = self.config.rel_mode,    #what kind of rel_rep generation algo to use 
            #rest are in kwargs
            hidden_size        = self.config.hidden_size, 
            layer_type         = self.config.projection_layer_type,
            ffn_ratio          = self.config.ffn_ratio,
            dropout            = self.config.dropout,
            no_context_rep     = self.config.rel_no_context_rep,   #how to handle edge case of no context tokens
            context_pooling    = self.config.rel_context_pooling,   #how to pool the context tokens
            window_size        = self.config.rel_window_size       
        )

        #filtering layer for relations
        self.rel_filter_head = FilteringLayer(self.config.hidden_size, 
                                              num_limit = self.config.num_limit,
                                              dropout   = self.config.dropout if self.config.filter_dropout else None) 

        # graph embedder
        self.graph_embedder = GraphEmbedder(self.config.hidden_size)
        
        #graph transformer layer
        self.graph_trans_layer = GraphTransformerModel(self.config.hidden_size,
                                                   num_heads  = self.config.num_heads,
                                                   num_layers = self.config.num_transformer_layers,
                                                   skip       = self.config.graph_skip_con)

        #used to calc the graph filter scores and graph loss
        GraphFilteringLayer = FilteringLayerBinaryDouble
        if self.config.graph_filter_head_type == 'single':
            GraphFilteringLayer = FilteringLayerBinarySingle
        
        self.graph_filter_head = GraphFilteringLayer(self.config.hidden_size, 
                                                     num_limit = self.config.num_limit,
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


    def transformer_and_lstm_encoder(self, x, transformer_encoder):
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
        result = transformer_encoder(x)
        #read in the values from result
        token_reps     = result['token_reps']       #embeddings of the word/sw tokens
        token_masks    = result['token_masks']      #masks for the word/sw tokens
        cls_reps       = result['cls_reps']         #embeddings for the CLS sw token, only if we are using HF
        #sw_span_ids    = result['sw_span_ids']      #tensor (batch, batch_max_seq_len*max_span_width, 2) => x['span_ids'] with values mapped using w2sw_map to the sw token start, end. Only if we are HF with no pooling.

        #Enrich token_reps
        #enriches the (batch, seq_len, hidden) shaped token reps with a bilstm to (batch, seq_len, hidden) shaped token reps, 
        #not clear if this is really necessary, but may be to deal with the side effects of first sw token pooling to get word tokens, 
        #at least that is what they allude to in their associated paper https://arxiv.org/pdf/2203.14710
        if self.config.use_lstm:
            token_reps = self.rnn(token_reps, token_masks) 

        return dict(token_reps      = token_reps, 
                    token_masks     = token_masks,
                    cls_reps        = cls_reps)         #if using HF


    def calc_top_k(self, masks, max_top_k):
        """
        Returns:
        top_k an int
        """
        #find the max available in any obs in the batch
        available = masks.sum(dim=1).max().item()
        #limit by max_top_k and return
        return int(min(available, max_top_k))


    def merge_maps(self, map_A_B, map_B_C):
        '''
        merge span_filter_maps if we have filtered the spans
        '''
        # Initialize the final mapping tensor (A->C) with -1 (indicating no mapping)
        map_A_C = torch.full_like(map_A_B, -1)
        # Identify indices in A->B that have valid mappings:
        # (1) They are not -1 (valid), and (2) The indices are within the bounds of B's second dimension
        valid_mask = (map_A_B != -1) & (map_A_B < map_B_C.shape[1])
        #Explicitly check if any valid index in map_A_B exceeds map_B_C dimension bounds
        if (map_A_B[valid_mask] >= map_B_C.shape[1]).any():
            self.config.logger.write(f"Warning: MapMerge issue: Index in map_A_B exceeds bounds of map_B_C dimension 1.", level='warning')
        # Temporarily set invalid indices in map_A_B to 0 to safely perform gather (-1 causes issues)
        safe_B_indices = map_A_B.masked_fill(~valid_mask, 0)
        # Gather the corresponding indices from map_B_C, for each valid position in map_A_B, find its corresponding C index
        gathered_C_indices = torch.gather(map_B_C, 1, safe_B_indices)
        # Update the output tensor map_A_C with valid indices gathered from map_B_C (Invalid indices (originally -1) remain unchanged)
        map_A_C[valid_mask] = gathered_C_indices[valid_mask]

        return map_A_C


    def prune_spans(self, filter_score_span, top_k_spans, span_filter_map_old=None):
        '''
        select the top K spans for the initial graph based on the span filter scores
        This is our smart neg sampling for spans
        #the top_k_spans is a tensor
        NOTE: it is possible in some cases with short sequences, that some of the shortlisted top_k_spans 
        are masked out, this is ok, it is captured in the span_masks, I'm just highlighting it
        For most cases however, the number of available spans far outnumber the max_top_k_spans
        '''
        _, num_spans = filter_score_span.shape

        #first sort the filter_score_span tensor descending
        #NOTE [1] is to get the idx as opposed to the values as torch.sort returns a tuple (vals, idx)
        sorted_span_idx = torch.sort(filter_score_span, dim=-1, descending=True)[1]
        
        '''
        select the top_k spans from each obs and form the span tensors (with the spans to use for the initial graph)
        This will create new tensors of length top_k_spans.shape[1] (dim 1) with the same order of span_idx as in the top_k_spans tensor
        NOTE: these candidate tensors are smaller than the span_rep tensors, so it saves memory!!!, otherwise, I do not see the reason fro doing this, you coudl literally, just pass the span_idx_to_keep
        '''
        span_idx_to_keep = sorted_span_idx[:, :top_k_spans]
        #make the mapping from span_ids idx to filtered span_ids idx using the span_idx_to_keep shortlist.
        span_filter_map = torch.full((self.batch, num_spans), -1, dtype=torch.long, device=self.device)
        # batch indices for gather
        #'''
        #old
        new_idx = torch.arange(top_k_spans, device=self.device).expand(self.batch, -1)  # Shape (batch, top_k_spans)
        span_filter_map[self.batch_ids, span_idx_to_keep] = new_idx   #shape (batch, num_spans_pre_filter)
        #'''
        '''
        #new
        batch_idx = torch.arange(batch, device=self.device).unsqueeze(1).expand_as(span_idx_to_keep)
        new_idx   = torch.arange(top_k_spans, device=self.device).unsqueeze(0).expand_as(span_idx_to_keep)
        span_filter_map[batch_idx, span_idx_to_keep] = new_idx
        '''

        #merge with old map if provided
        if span_filter_map_old is not None:
            span_filter_map = self.merge_maps(span_filter_map_old, span_filter_map)

        return span_idx_to_keep, span_filter_map



    def filter_spans(self, filter_score_span, max_top_k_spans, span_filter_map, span_ids, span_masks, span_labels=None):
        '''
        this does the span filtering from the filter scores

        it outputs:
        - filtered span_ids, masks and labels
        - top_k_spans that was calculated and used
        - span_filter_map => maps pre-filter span_ids to post-filter span ids
        - the span_idx from the pre-filter span_ids used to make the post-filter span_ids
        '''
        #top_k_spans = self.calc_top_k_spans_old(filter_score_span, max_top_k_spans)
        top_k_spans = self.calc_top_k(span_masks, max_top_k_spans)
        #abort the forward run if there is an obs with no candidate spans
        #need to think about why this may happen, I think it is ok, it is just when the model is untrained
        #print(f'top_k_spans: {top_k_spans}')
        if top_k_spans == 0 or filter_score_span.shape[1] == 0: 
            return 1

        #prune the spans to top_k_spans
        span_idx_to_keep, span_filter_map = self.prune_spans(filter_score_span, top_k_spans, span_filter_map) 

        #filter the reps, masks and labels
        span_ids = span_ids[self.batch_ids, span_idx_to_keep, :]  # shape: (batch, top_k_spans, 2)
        span_masks = span_masks[self.batch_ids, span_idx_to_keep]    # shape: (batch, top_k_spans)
        if span_labels is not None:
            span_labels = span_labels[self.batch_ids, span_idx_to_keep]  # shape: (batch, top_k_spans) unilabels or (batch, top_k_spans, num_span_types) multilabels
        
        return dict(span_ids         = span_ids,
                    span_masks       = span_masks,
                    span_labels      = span_labels,
                    top_k_spans      = top_k_spans,
                    span_idx_to_keep = span_idx_to_keep,
                    span_filter_map  = span_filter_map)



    def prune_rels(self, filter_score_rel, top_k_rels):
        '''
        filter the rels down to top_k_rels with the rel filter scores
        This is our smart neg sampling for rels
        NOTE: it is possible in some cases with short sequences, that some of the shortlisted top_k_rels 
        are masked out, this is ok, it is captured in the rel_masks, I'm just highlighting it
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


    def set_force_pos(self, type):
        '''
        This determines if force pos cases is True/False dependent on the configs and the current step and the filter type
        '''
        #Test if TF needs to be disabled
        if (not self.training and self.config.span_force_pos != 'always-eval') or \
            self.config.span_force_pos == 'never':   #in this case we have completely disabled TF in span and rel modes
            return False
                
        #do the training scenarios (or the case where span force pos is always-eval)
        if type == 'span':
            if self.config.span_force_pos == 'temp':
                if self.step+1 > self.config.force_pos_step_limit:
                    return False
                else:
                    return True
            return True

        if type == 'rel':
            if self.config.rel_force_pos == 'never':
                return False
            else:    #if it is 'follow'
                if self.config.span_force_pos == 'temp':
                    if self.step+1 > self.config.force_pos_step_limit:
                        return False
                    else:
                        return True
                return True

        return False   
 

    def calc_pred_losses(self, logits_span, span_labels, span_masks, 
                               logits_rel, rel_labels, rel_masks):
        #span pred loss
        pred_span_loss = classification_loss(logits_span, 
                                             span_labels, 
                                             span_masks, 
                                             reduction     = self.config.loss_reduction, 
                                             label_type    = 'unilabel')

        #rel pred loss
        pred_rel_loss = classification_loss(logits_rel, 
                                            rel_labels, 
                                            rel_masks, 
                                            reduction     = self.config.loss_reduction, 
                                            label_type    = self.config.rel_labels)
        return pred_span_loss, pred_rel_loss
    

    def neg_sampler(self, masks, binary_labels=None, neg_sampling_limit=100):
        '''
        Creates a new span/rel mask with only a subset of the negative samples.
        NOTE: Only applicable during training.
        '''
        if binary_labels is None:
            return masks

        batch, _ = masks.shape

        #Initialize new mask with all False
        new_masks = torch.zeros_like(masks, dtype=torch.bool)
        #set pos cases to True in the new mask
        new_masks[binary_labels] = True  
        #Process each batch observation separately
        for b in range(batch):  # Iterate over batch
            #Find indices of negative cases where mask is True
            neg_indices = ((binary_labels[b] == False) & (masks[b] == True)).nonzero(as_tuple=False).squeeze(-1)
            #Select up to `neg_sampling_limit` negatives
            if neg_indices.numel() > 0:
                num_neg_samples = min(neg_sampling_limit, neg_indices.shape[0])
                selected_neg_indices = neg_indices[torch.randperm(neg_indices.shape[0])[:num_neg_samples]]
                new_masks[b, selected_neg_indices] = True  # Activate selected negatives

        return new_masks



    def neg_sampler_old(self, masks, binary_labels=None, neg_sampling_limit=100):
        '''
        Creates a new span/rel mask with only a subset of the negative samples.
        NOTE: Only applicable during training.
        '''
        if binary_labels is None:
            return masks

        #Initialize new mask with all False
        new_masks = torch.zeros_like(masks, dtype=torch.bool)
        #set pos cases to True in the new mask
        new_masks[binary_labels] = True  
        #Determine the number of positive samples per batch
        num_pos_cases = torch.sum(binary_labels, dim=1)  # Shape: (batch,)
        max_neg_samples = neg_sampling_limit - num_pos_cases  #(batch,)

        #Process each batch obs separately
        for i in range(masks.shape[0]):  # Iterate over the batch dimension
            #Find indices of negative cases where the mask is True
            #neg_indices = ((span_binary_labels[i] == False) & (span_masks[i] == True)).nonzero(as_tuple=False)
            neg_indices = ((binary_labels[i] == False) & (masks[i] == True)).nonzero(as_tuple=False).squeeze(-1)
            #If there are negative cases, perform neg sampling
            if neg_indices.numel() > 0:
                num_neg_samples = min(max_neg_samples[i], neg_indices.shape[0])
                selected_neg_indices = neg_indices[torch.randperm(neg_indices.shape[0])[:num_neg_samples]]
                #Activate the selected negatives
                new_masks[i, selected_neg_indices] = True

        return new_masks


    def post_neg_sample_pruning(self, ids, masks, labels):
        """
        Prunes masked-out sapn/rel elements and repads the tensors to save space
        
        Args:
            ids (torch.Tensor): Shape (b, num_items, 2)
            labels (torch.Tensor): Shape (b, num_items, num_classes) (multi-label) or (b, num_items) (uni-label).
            masks (torch.Tensor): Shape (b, num_items), boolean mask of valid relations.

        Returns:
            dict: New pruned and padded tensors for `ids`, `labels`, and `masks`.

        NOTE: no need to check for labels as this will only ever be run during training with labels and neg sampling
        """
        batch_size = masks.shape[0]
        is_multilabel = len(labels.shape) == 3 # Detect multi-label case
        pruned_ids, pruned_masks, pruned_labels = [], [], []

        for i in range(batch_size):
            # Get valid indices for this batch
            valid_indices = masks[i].nonzero(as_tuple=True)[0]  # 1D indices of kept relations

            # Subset tensors using valid indices
            pruned_ids.append(ids[i, valid_indices])  # (num_valid_items, 2)
            pruned_masks.append(torch.ones_like(valid_indices, dtype=torch.bool))  # (num_valid_items,)
            pruned_labels.append(labels[i, valid_indices])  # (num_valid_items, num_classes) or (num_valid_items,)

        # Pad tensors back to uniform batch shape
        padded_ids = pad_sequence(pruned_ids, batch_first=True, padding_value=0)  # (b, max_valid_items, 2)
        padded_masks = pad_sequence(pruned_masks, batch_first=True, padding_value=False)  # (b, max_valid_items)
        padded_labels = pad_sequence(pruned_labels, batch_first=True, padding_value=0)  # (b, max_valid_items, num_classes) or (b, max_valid_items)

        return {
            "ids": padded_ids,
            "masks": padded_masks,
            "labels": padded_labels
        }



    def make_span_label_data(self, token_reps, cls_reps, span_annotations, neg_limit, alpha):
        """
        Constructs span label tensors and computes their representations.

        This function extracts gold (positive) span IDs from the full list of candidate spans using
        the original annotation indices and mapping (orig_map). It computes the span representations
        for these gold spans using the configured span representation layer.

        Args:
            token_reps (Tensor): Token-level hidden states from the encoder. Shape: (batch, seq_len, hidden).
            all_span_ids (Tensor): All candidate span IDs. Shape: (batch, total_spans, 2).
            cls_reps (Tensor): Optional CLS representations for each input. Shape: (batch, hidden).
            span_annotations (List[List[Tuple[int, int, int]]]): Raw gold span annotations per instance.
                Each tuple is (start, end, label).
            orig_map (List[Dict[int, int]]): Maps each annotation index to its corresponding index in all_span_ids.
            neg_limit (float): Negative masking value for invalid span positions.
            alpha (float): Alpha factor for dynamic span pooling window calculation.

        Returns:
            label_span_ids (Tensor): Extracted gold span IDs. Shape: (batch, max_gold_spans, 2).
            label_span_masks (BoolTensor): Mask indicating valid label spans. Shape: (batch, max_gold_spans).
            label_span_reps (Tensor): Span representations computed for gold spans. Shape: (batch, max_gold_spans, hidden).
        """
        batch = len(span_annotations)
        device = token_reps.device

        max_labels = max(len(spans) for spans in span_annotations) or 1  # handle edge case

        #label span ids are annotation aligned, pos case only
        label_span_ids = torch.zeros(batch, max_labels, 2, dtype=torch.long, device=device)     # (b, max_gold, 2)
        label_span_masks = torch.zeros(batch, max_labels, dtype=torch.bool, device=device)
        label_span_labels = torch.zeros(batch, max_labels, dtype=torch.long, device=device)
        for b in range(batch):
            for i, (start, end, label) in enumerate(span_annotations[b]):
                label_span_ids[b, i, 0] = start
                label_span_ids[b, i, 1] = end
                label_span_masks[b, i] = 1
                label_span_labels[b, i] = self.config.s_to_id[label]

        label_widths = label_span_ids[:,:,1] - label_span_ids[:,:,0]
        label_span_reps = self.span_rep_layer(token_reps, 
                                              span_ids    = label_span_ids, 
                                              span_masks  = label_span_masks, 
                                              cls_reps    = cls_reps,
                                              span_widths = label_widths,
                                              neg_limit   = neg_limit,
                                              alpha       = alpha)

        return label_span_ids, label_span_masks, label_span_labels, label_span_reps




    def make_rel_label_data(self, token_reps, token_masks, label_span_reps, label_span_ids, rel_annotations, neg_limit, cls_reps=None):
        """
        Constructs label_rel_ids, label_rel_masks, label_rel_reps tensors

        desc here...
        
                
        Returns:
            label_span_ids (Tensor): Extracted gold span IDs. Shape: (batch, max_gold_spans, 2).
            label_span_masks (BoolTensor): Mask indicating valid label spans. Shape: (batch, max_gold_spans).
            label_span_reps (Tensor): Span representations computed for gold spans. Shape: (batch, max_gold_spans, hidden).
        """
        device = label_span_reps.device
        batch = len(rel_annotations)
        max_labels = max(len(rels) for rels in rel_annotations) or 1  # handle edge case

        #fill the label_rel_ids with the raw head, tail ids, i.e. annotation aligned pos case only
        label_rel_ids = torch.zeros(batch, max_labels, 2, dtype=torch.long, device=device)     # (b, max_labels, 2)
        label_rel_masks = torch.zeros(batch, max_labels, dtype=torch.bool, device=device)
        if self.config.rel_labels == 'multilabel':
            label_rel_labels = torch.zeros(batch, max_labels, self.config.num_rel_types, dtype=torch.bool, device=device)
        else:
            label_rel_labels = torch.zeros(batch, max_labels, dtype=torch.long, device=device)
        for b in range(batch):
            for i, (head, tail, label) in enumerate(rel_annotations[b]):
                label_rel_ids[b, i, 0] = head
                label_rel_ids[b, i, 1] = tail
                label_rel_masks[b, i] = 1
                if self.config.rel_labels == 'multilabel':
                    label_rel_labels[b, i, self.config.r_to_id[label]] = True
                else:
                    label_rel_labels[b, i] = self.config.r_to_id[label]

        if self.rel_rep_layer == 'no_context':
            label_rel_reps = self.rel_rep_layer(span_reps = label_span_reps,           #annotation aligned span reps for pos cases only   
                                                rel_ids   = label_rel_ids)             #aligned to label_span_reps/ids
        else:
            label_rel_reps = self.rel_rep_layer(span_reps       = label_span_reps,     #annotation aligned span reps for pos cases only   
                                                span_ids        = label_span_ids,      #annotation aligned span ids for pos cases only   
                                                rel_ids         = label_rel_ids,       #annotation aligned rel ids for pos cases only   
                                                token_reps      = token_reps, 
                                                token_masks     = token_masks,
                                                rel_masks       = label_rel_masks,     #annotation aligned rel masks for pos cases only   
                                                neg_limit       = neg_limit,
                                                cls_reps        = None)     #not used

        return label_rel_ids, label_rel_masks, label_rel_labels, label_rel_reps



    def prune_hanging_rels(self, rel_ids, logits_span, span_masks, rel_masks):
        """
        Remove any relation whose head or tail span was either
        predicted negative or has been masked out by span pruning.

        Args:
            rel_ids     (B, R, 2): head/tail span indices
            logits_span (B, K, C): raw span‑classification logits
            span_masks  (B, K):    mask of current valid spans
            rel_masks   (B, R):    current mask of relations

        Returns:
            new_rel_masks  (B, R): updated, with hanging rels cleared
            dropped_rels   (B, R): True where we just dropped them
        """
        span_probs = torch.softmax(logits_span, dim=-1)        # (B, K, num_span_types)
        span_preds = span_probs.argmax(dim=-1)                 # (B, K)  ← class index per span
        is_pos_span = (span_preds != 0)                        # (B, K)  ← True for any non‑neg class
        valid_span = is_pos_span & span_masks                  # (B, K)
        
        head_idx = rel_ids[..., 0]   # (B, R)
        tail_idx = rel_ids[..., 1]   # (B, R)
        batch_ar = torch.arange(valid_span.size(0), device=valid_span.device).unsqueeze(1).expand_as(head_idx)
        #look up which of those spans were predicted positive
        head_ok = valid_span[batch_ar, head_idx]          # (B, R)
        tail_ok = valid_span[batch_ar, tail_idx]          # (B, R)
        #only keep relations where both ends survived
        keep_rel = head_ok & tail_ok                  # (B, R)
        new_rel_masks = rel_masks & keep_rel
        hanging_masks = rel_masks & ~keep_rel

        return new_rel_masks, hanging_masks



    def prune_redundant_spans(self, span_ids, logits_span, span_labels, span_masks, overlap_thd=0.75):
        """
        Greedy span suppression based only on IoU overlap.

        Args:
            span_ids (B,K,2): span boundaries [start, end+1]
            logits_span (B,K,C): raw span logits
            span_labels (B,K): 0=neg, >0=gold
            span_masks (B,K): valid span mask
            overlap_thd (float): IoU threshold to suppress (default 0.85)

        Returns:
            new_span_masks (B,K): kept spans
            redundant_mask (B,K): dropped spans
        """
        B, K, _ = span_ids.shape

        new_masks = torch.zeros_like(span_masks, dtype=torch.bool)
        redundant = torch.zeros_like(span_masks, dtype=torch.bool)

        for b in range(B):
            starts_ends = span_ids[b].float()  # (K,2)
            valid = span_masks[b].bool()       # (K,)
            gold = (span_labels[b] > 0) & valid

            if valid.sum() <= 1:
                new_masks[b] = valid
                continue

            # Compute per-span confidence (ignoring class 0)
            probs = torch.softmax(logits_span[b], dim=-1)  # (K,C)
            preds = probs.argmax(dim=-1)                  # (K,)
            avail = valid & (preds != 0)

            if avail.sum() == 0:
                continue

            if probs.size(-1) > 1:
                conf = probs[:, 1:].max(dim=-1).values
            else:
                conf = probs.squeeze(-1)
            conf[~avail] = -self.config.num_limit

            # Boost gold+predicted-positive spans to sort first
            gold_pred_pos = gold & avail
            if gold_pred_pos.any():
                max_conf = conf[avail].max().item()
                conf[gold_pred_pos] = max_conf + 1.0

            # Sort spans by descending confidence
            order = torch.argsort(conf, descending=True)
            kept = []

            for i in order:
                if not avail[i]:
                    continue

                si, ei = starts_ends[i]
                should_suppress = False

                for j in kept:
                    sj, ej = starts_ends[j]
                    inter = max(0.0, min(ei, ej) - max(si, sj))
                    union = max(1.0, max(ei, ej) - min(si, sj))
                    iou = inter / union

                    if iou >= overlap_thd:
                        # Prefer the longer span: if current span is shorter, suppress it
                        len_i = ei - si
                        len_j = ej - sj
                        if len_i < len_j:
                            should_suppress = True
                        elif len_i == len_j and conf[i] <= conf[j]:
                            should_suppress = True
                        break

                if should_suppress:
                    redundant[b, i] = True
                else:
                    kept.append(i)

            new_masks[b, kept] = True
            redundant[b, gold_pred_pos] = False  # Never suppress gold

        return new_masks, redundant



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
        run_type_is_train = self.config.run_type == 'train'
        self.step = step
                 
        #Run the transformer encoder and lstm encoder layers
        result = self.transformer_and_lstm_encoder(x, self.transformer_encoder_span)
        #read in data from results or x
        tokens          = x['tokens']                   #the list of raged lists of word tokens
        token_reps      = result['token_reps']          #(batch, batch_max_seq_len, hidden) => float, sw or w token aligned depedent pooling   
        token_masks     = result['token_masks']         #(batch, batch_max_seq_len) => bool, sw or w token aligned depedent pooling   
        span_ids        = x['span_ids']                 #(batch, batch_max_seq_len * max_span_width, 2) => int, w aligned span_ids
        span_masks      = x['span_masks']               #(batch, batch_max_seq_len * max_span_width) => bool, includes the spans that are pos cases and selected neg cases for each batch obs
        span_labels     = x['span_labels']              #(batch, batch_max_seq_len * max_span_width) int
        num_span_types  = self.config.num_span_types    #scalar, includes the none type (idx 0) for unilabel and only pos types for multilabel
        num_rel_types   = self.config.num_rel_types     #scalar, includes the none type (idx 0) for unilabel and only pos types for multilabel
        cls_reps        = result['cls_reps']            #(batch,hidden)
        orig_map        = x['orig_map']                 #list of dicts for the mapping of the orig annotation span list id to the span_ids dim 1 idx
        rel_annotations  = x['relations']               #list of ragged list of tuples => the positive cases for each obs  => list of empty lists if no labels
        span_annotations = x['spans']                   #list of ragged list of tuples => the positive cases for each obs  => list of empty lists if no labels

        #set some initial values
        self.device = token_reps.device
        self.batch, self.batch_max_seq_len, _ = token_reps.shape
        self.batch_ids = torch.arange(self.batch, dtype=torch.long, device=self.device).unsqueeze(-1)  # shape: (batch, 1)
        span_filter_map = None   #start off with no span_filter_map
        all_span_ids = span_ids.clone()    #need this for later
        #init losses
        tagger_loss       = torch.tensor(0., device=self.device)  # scalar tensor
        filter_loss_span  = torch.tensor(0., device=self.device)  # scalar tensor
        filter_loss_rel   = torch.tensor(0., device=self.device)  # scalar tensor
        filter_loss_graph = torch.tensor(0., device=self.device)  # scalar tensor
        lost_rel_loss     = torch.tensor(0., device=self.device)  # scalar tensor
        prune_loss        = torch.tensor(0., device=self.device)  # scalar tensor
        token_logits = None
        span_logits = None

        #######################################################
        # SPANS ###############################################
        #######################################################
        #TOKEN TAGGER HEAD AND FILTERING
        if self.config.token_tagger:
            #Token Tagger
            ##########################
            #run the token tagger filter here to cut down span_ids to a smaller set of span_ids to process (much smaller)
            #NOTE: there is no sense in doing any neg sampling for training with token tagging as the token tagging labels and loss are based only on the pos cases
            #NOTE: for token tagging, we do not support no token pooling, i.e. you must pool bert subtokens back to word tokens!!!  Why because I said so, It is too messy otherwise
            #the inputs will be the token_reps, token_masks, span_ids, span_masks and span_labels
            #the outputs will be the new span_ids, span_masks, span_labels as well as the token_tagger_loss (to replace the span_filter_loss or add to it)
            #NOTE: the token_tagger_loss will NOT have teacher forcing, but the output span_ids/labels/masks will have teacher forcing for training so pos cases are passed
            result = self.token_tag_layer(token_reps      = token_reps,
                                          token_masks     = token_masks,
                                          span_ids        = span_ids,     
                                          span_masks      = span_masks, 
                                          span_labels     = span_labels,    #will be None for predict
                                          force_pos_cases = self.set_force_pos('span'),   
                                          reduction       = self.config.loss_reduction)
            #read in data from results
            #NOTE: we overwrite the span_ids, span_masks and span_labels to the new filtered down set
            span_ids    = result['out_span_ids']            #(batch, batch_max_num_filtered_spans, 2) => int
            span_masks  = result['out_span_masks']          #(batch, batch_max_num_filtered_spans) => bool
            span_labels = result['out_span_labels']         #(batch, batch_max_num_filtered_spans) => int
            filter_score_span = result['out_span_scores']   #(batch, batch_max_num_filtered_spans) => float
            tagger_loss = result['tagger_loss']             #single value tensor
            span_filter_map = result['span_filter_map']     #mapping from old span_ids to new span_ids dim 1 idx => tensor    (batch, all_possible_spans) => int
            token_logits = result['token_logits']
            #print(f'\nnew num spans: {span_masks.shape[1]}')

            #use the tagger filter scores to filter spans
            result = self.filter_spans(filter_score_span = filter_score_span,
                                       max_top_k_spans   = self.config.max_top_k_spans_pre if self.config.span_filtering_type == 'both' else self.config.max_top_k_spans,
                                       span_filter_map   = span_filter_map,   #span filter_map will not be None as it was updated by the token tagger
                                       span_ids          = span_ids,
                                       span_masks        = span_masks,
                                       span_labels       = span_labels if run_type_is_train else None)
            #read in results
            if not isinstance(result, dict):
                return result
            span_ids         = result['span_ids']
            span_masks       = result['span_masks']
            span_labels      = result['span_labels']
            span_filter_map  = result['span_filter_map']    #the updated filter map after filtering
            #top_k_spans      = result['top_k_spans']    #dont need here
            #span_idx_to_keep = result['span_idx_to_keep']   #dont need here
        ############################
        #end of token tagger
        ############################

        #NEG SAMPLING
        #NOTE: if we neg sample spans, only do it here, it makes no sense to do it before the token tagger
        #NOTE: neg sampling is only applied to training and always passes the pos cases
        if self.config.span_neg_sampling and self.training:
            #does neg sampling for the training case
            span_masks = self.neg_sampler(span_masks, self.binarize_labels(span_labels), self.config.span_neg_sampling_limit)


        #MAKE SPAN REPS
        #generate the span reps from the token reps and the span start/end idx and outputs a tensor of shape (batch, num_spans, hidden), 
        #num_spans will be less if the token tagger filtering was done first or all possible if not (max_seq_len * max_span_width)
        span_reps = self.span_rep_layer(token_reps, 
                                        span_ids    = span_ids, 
                                        span_masks  = span_masks, 
                                        cls_reps    = cls_reps,
                                        span_widths = span_ids[:,:,1] - span_ids[:,:,0],
                                        neg_limit   = -self.config.num_limit,
                                        alpha       = self.config.span_win_alpha)

        #BINARY FILTER HEAD AND FILTERING
        if self.config.span_filtering_type in ['bfhs', 'both']:   
            #use the binary filter head on the span reps to make the filter score and filter loss
            filter_score_span, filter_loss_span = self.span_filter_head(span_reps, 
                                                                        span_masks, 
                                                                        self.binarize_labels(span_labels) if run_type_is_train else None, 
                                                                        force_pos_cases = self.set_force_pos('span'),
                                                                        reduction       = self.config.loss_reduction,
                                                                        loss_only       = False)           

            #use the filter scores to filter spans
            result = self.filter_spans(filter_score_span = filter_score_span,
                                       max_top_k_spans   = self.config.max_top_k_spans,
                                       span_filter_map   = span_filter_map,    #will be None if no token tagger
                                       span_ids          = span_ids,
                                       span_masks        = span_masks,
                                       span_labels       = span_labels if run_type_is_train else None)
            #read in results
            if not isinstance(result, dict):
                return result
            span_ids         = result['span_ids']
            span_masks       = result['span_masks']
            span_labels      = result['span_labels']
            top_k_spans      = result['top_k_spans']
            span_idx_to_keep = result['span_idx_to_keep']
            span_filter_map  = result['span_filter_map']
            
            #filter the span_reps
            span_reps = span_reps[self.batch_ids, span_idx_to_keep]   # shape: (batch, top_k_spans, hidden)
        ############################
        #END OF BINARY FILTER HEAD
        ############################

        ###############################################################
        # RELATIONS ###################################################
        ###############################################################
        #get the rel labels from x['relations'] with dim 1 in same order as span_id dim 1 expanded to top_k_spans**2 (this is why we need to limit top_k_spans to as low as possible)
        #NOTE: the rel_masks have the diagonal set to 0, i.e self relations masked out
        #this returns rel_labels of shape (batch, top_k_spans**2) int for unilabel or (batch, top_k_spans**2, num span types) bool for multilabel => None if not run_type_is_train
        #as well as rel_masks, rel_ids of shape (batch, top_k_spans**2), (batch, top_k_spans**2, 2) respectively
        #if run_type_is_train, it also returns the lost_rel_cnts tensor of shape (batch) which is one int per batch obs indicating the number fo rels that had annotations but did not make it to get rel_ids as they were filtered out byt the span filtering
        #we will add a lost rel penalty loss for this
        rel_ids, rel_masks, rel_labels, lost_rel_counts, lost_rel_penalty = self.rel_processor.get_rel_tensors(all_span_ids, 
                                                                                                               span_masks,            #the span mask for each selected span  (batch, top_k_spans)
                                                                                                               rel_annotations,       #the raw relation annotations data.  NOTE: will be None for no labels
                                                                                                               orig_map,              #maps annotation span list id to the dim 1 id in all_span_ids.  NOTE: will be None for no labels  
                                                                                                               span_filter_map,       #maps all_span_ids to current span_ids which will be different after token tagging heads and filtering
                                                                                                               token_logits,          #will be none if not tths or both
                                                                                                               filter_score_span)     #will be used if in bfhs mode
        if self.config.rel_neg_sampling and self.training:
            #does neg sampling for the training case
            #NOTE: check if strong neg sampling is even possible for this particular model structure, my feeling is not
            #NOTE: neg sampling is only applied to training and always passes the pos cases
            rel_masks = self.neg_sampler(rel_masks, self.binarize_labels(rel_labels), self.config.rel_neg_sampling_limit)

        '''
        Make the relation reps, has several options, based on config.rel_mode:
        1) no_context => graphER => juts concat the head and tail span reps
        2) between_context => concatenate the head and tail reps with between spans rep for context => to be coded
        3) window_context => concatenate the head and tail reps with windowed context reps, i.e. window context takes token reps in a window before and after each span and attention_pools, max_pool them => to be coded
        4+) working on more options, see code for details
        '''
        if not self.config.bert_shared_unmarked_span_rel:
            #get a separate set of token and span reps for the rels from the rel bert
            result = self.transformer_and_lstm_encoder(x, self.transformer_encoder_rel)
            token_reps = result['token_reps']          #(batch, batch_max_seq_len, hidden) => float, sw or w token aligned depedent pooling   
            cls_reps  = result['cls_reps']            #(batch,hidden)
            
            #remake the span_reps for use in building the rel reps
            span_reps = self.span_rep_layer(token_reps, 
                                            span_ids    = span_ids, 
                                            span_masks  = span_masks, 
                                            cls_reps    = cls_reps,
                                            span_widths = span_ids[:,:,1] - span_ids[:,:,0],
                                            neg_limit   = -self.config.num_limit,
                                            alpha       = self.config.span_win_alpha)

        #make the special span reps for use in making the rel reps
        if self.config.modified_span_reps_for_rel_reps:
            #remake the special span_reps for use in building the rel reps
            # basically leave out the cls reps and the width embeddings
            #this just follows the idea from spert which only left ou the cls reps
            span_reps_rel = self.span_rep_layer_for_rels(token_reps, 
                                                         span_ids    = span_ids, 
                                                         span_masks  = span_masks, 
                                                         cls_reps    = None,
                                                         span_widths = None, #span_ids[:,:,1] - span_ids[:,:,0],
                                                         neg_limit   = -self.config.num_limit,
                                                         alpha       = self.config.span_win_alpha)
        else:
            span_reps_rel = span_reps


        #output shape (batch, num_spans**2, hidden)   
        #NOTE: if we used neg sampling on the rels and pruned the tensors, then the dim length will be much shorter than top_k_spans**2
        #NOTE: this is a temporary if statement, I have not updated the context based rel rep code yet
        #if self.rel_rep_layer.rel_mode == 'no_context':
        #    rel_reps = self.rel_rep_layer(span_reps = span_reps,           #top_k_span aligned span_reps 
        #                                  rel_ids   = rel_ids)             #aligned to span_reps/ids
        #else:
        rel_reps = self.rel_rep_layer(token_reps      = token_reps, 
                                      token_masks     = token_masks,
                                      span_reps       = span_reps_rel,     #top_k_span aligned span_reps 
                                      span_ids        = span_ids,      #top_k_span aligned span_ids
                                      rel_ids         = rel_ids,       #aligned to span_reps/ids
                                      rel_masks       = rel_masks,
                                      neg_limit       = -self.config.num_limit,
                                      cls_reps        = None) #cls_reps)   

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
        filter_score_rel, filter_loss_rel = self.rel_filter_head(rel_reps, 
                                                                 rel_masks, 
                                                                 self.binarize_labels(rel_labels) if run_type_is_train else None, 
                                                                 force_pos_cases = self.set_force_pos('rel'),
                                                                 reduction       = self.config.loss_reduction,
                                                                 loss_only       = False)           

        #Calculate the number of rels to shortlist
        #top_k_rels = self.calc_top_k_rels_old(filter_score_rel, self.config.max_top_k_rels)   #returns a scalar which should be << top_k_spans**2
        top_k_rels = self.calc_top_k(rel_masks, self.config.max_top_k_rels)   #returns a scalar which should be << top_k_spans**2
        #print(f'top_k_rels: {top_k_rels}')
        #abort the forward run if there are no candidate rels
        if top_k_rels == 0 or filter_score_rel.shape[1] == 0:
            return 2
        #prune the rels
        rel_idx_to_keep = self.prune_rels(filter_score_rel, top_k_rels)
        rel_reps   = rel_reps[self.batch_ids, rel_idx_to_keep]    #(batch, top_k_rels, hidden)  float
        rel_masks  = rel_masks[self.batch_ids, rel_idx_to_keep]      #(batch, top_k_rels)  bool
        rel_ids    = rel_ids[self.batch_ids, rel_idx_to_keep]     #(batch, top_k_rels, 2) int
        rel_labels = rel_labels[self.batch_ids, rel_idx_to_keep] if run_type_is_train else None     #(batch, top_k_rels) int for unilabel or (batch, top_k_rels, num rel types) int for multilabel 

        #########################################################
        #the graph transformer section
        #########################################################
        if not self.config.use_graph:    #if we bypass the graph completely
            node_reps, edge_reps = span_reps, rel_reps
        
        else:    #use the graph transformer
            #Generate the node and edge reps
            #this basically superimposes a node identifier to the span_reps and an edge identifier to the rel_reps
            #i.e. the id is added by element wise addition to the reps, the same way as pos encoding is added
            #this is preparing the reps for graph attention
            #node reps with be shape (batch, top_k_spans, hidden)
            #edge reps with be shape (batch, tp_k_rels, hidden)
            node_reps, edge_reps = self.graph_embedder(span_reps, rel_reps, span_masks, rel_masks)

            #merge the node and edge reps into one tensor so we can run it through a attention block
            #think of it like a sequence and each span or pair is a token, so he is concat the spans and pairs
            #so shape will be (batch, top_k_spans + top_k_rels, hidden)
            graph_reps = torch.cat((node_reps, edge_reps), dim=1)   #shape (batch, top_k_spans + top_k_rels, hidden)   float
            #store the node and edge cnts for splitting later
            node_cnt, edge_cnt = node_reps.shape[1], edge_reps.shape[1]    #top_k_spans and top_k_rels
            #merge masks
            graph_masks = torch.cat((span_masks, rel_masks), dim=1)   #(batch, top_k_spans + top_k_rels)   bool
            #binarize and merge labels as this is just used for graph pruning
            binary_graph_labels = torch.cat((self.binarize_labels(span_labels), self.binarize_labels(rel_labels)), dim=1) if run_type_is_train else None   #(batch, top_k_spans + top_k_rels) int (0,1) for unilabel/multilabel as they have been binarized

            ###################################################
            #Apply transformer layer and keep_head
            #Q: why not use a GAT => beacuse the GAT doesn't enrich the edge reps, only the node reps, even if we get it to take in the edge reps, it is garbage
            ###################################################
            #this is a using the torch built in mha transformer encoder, mask is the key padding mask
            #shape out will be same as input reps (batch, top_k_spans + top_k_rels, hidden)
            graph_reps = self.graph_trans_layer(graph_reps, graph_masks)
            #run graph reps through a binary classification head
            #output will be shapes:  None, scalar (None if run_type_is_train is False)
            _, filter_loss_graph = self.graph_filter_head(graph_reps, 
                                                          graph_masks,
                                                          binary_graph_labels,    #will be None if run_type_is_train is False
                                                          force_pos_cases = False,   #should never be True as filter scores are not used for the Graph filter, only the loss 
                                                          reduction       = self.config.loss_reduction,
                                                          loss_only       = True)           

            #Split the reps back to nodes(spans) and edges(rels)
            node_reps, edge_reps = graph_reps.split([node_cnt, edge_cnt], dim=1)                             #(batch, top_k_spans + top_k_rels, hidden) => (batch, top_k_spans, hidden), (batch, top_k_rels, hidden)
            
            #set the node and edge reps back to the pre-graph transformer span_reps and rel_reps if use_graph_reps is False
            #in this case only the graph loss is used from teh graph processing
            if not self.config.use_graph_reps:
                node_reps, edge_reps = span_reps, rel_reps



        ############################################################################
        #Output Heads
        ############################################################################
        ############################################################################
        #NOTE: num_span_types and num_rel_types are the number of neg/pos types for unilabel and pos types only for multilabel
        logits_span = self.output_head_span(node_reps)  # Shape: (batch, top_k_spans, num_span_types)
        logits_rel = self.output_head_rel(edge_reps)   # Shape: (batch, top_k_rels, num_rel_types)

        ########################################################################
        #all this stuff is not really critical
        #do the post model pruning
        #this is for the training with the train set case where we are calculating gradients
        if self.config.consistency_penalties_train and self.training and self.step > self.config.penalty_step_limit:
            #remove redundant spans and make the penalty
            span_masks_mod, redundant_masks = self.prune_redundant_spans(span_ids, logits_span, span_labels, span_masks, overlap_thd=self.config.overlap_thd)
            span_scores = torch.sigmoid(logits_span.squeeze(-1))   #unilabel scores
            redundant_span_penalty = span_scores[redundant_masks].sum()
            #remove hanging rels and make the penalty
            _, hanging_masks = self.prune_hanging_rels(rel_ids, logits_span, span_masks_mod, rel_masks)
            rel_scores = torch.sigmoid(logits_rel.max(dim=-1).values)  # (B, R)    #best multilabel score
            hanging_rel_penalty = rel_scores[hanging_masks].sum()

            prune_loss = prune_loss + self.config.hanging_rel_alpha * hanging_rel_penalty
            prune_loss = prune_loss + self.config.redundant_span_alpha * redundant_span_penalty


        #deal with pruning for the predict inference case
        #best to do this externally  
        if self.config.post_model_prune_inference and not run_type_is_train:
            #remove redundant spans
            empty_labels = torch.zeros_like(span_masks, dtype=torch.int)
            span_masks, _ = self.prune_redundant_spans(span_ids, logits_span, empty_labels, span_masks, overlap_thd=self.config.overlap_thd)
            #remove hanging rels
            rel_masks, _ = self.prune_hanging_rels(rel_ids, logits_span, span_masks, rel_masks)
        ########################################################################


        #final processing and return
        total_loss = None
        loss_breakdown = None
        #########################################################
        #do loss if we have labels
        #########################################################
        if run_type_is_train:
            #Compute losses for spans and rels final classifier heads using the span/rel type reps
            #NOTE: uses CELoss for unilabels and BCELoss for multilabels
            pred_loss_span, pred_loss_rel = self.calc_pred_losses(logits_span,
                                                                  span_labels, 
                                                                  span_masks, 
                                                                  logits_rel,
                                                                  rel_labels, 
                                                                  rel_masks)

            #get the lost rel_loss (only will have values if we are using non teacher forcing for spans, i.e. temp and past the warmup)
            lost_rel_loss = lost_rel_penalty * self.config.lost_rel_alpha
            #self.config.logger.write(f'lost rel count: {lost_rel_counts.sum().item()}')

            #get the total loss
            total_loss = sum([tagger_loss, filter_loss_span, filter_loss_rel, lost_rel_loss, filter_loss_graph, prune_loss, self.config.span_loss_mf*pred_loss_span, self.config.rel_loss_mf*pred_loss_rel])
            loss_breakdown = dict(step              = step,
                                  tagger_loss       = tagger_loss.detach().cpu().item(),
                                  filter_loss_span  = filter_loss_span.detach().cpu().item(),
                                  filter_loss_rel   = filter_loss_rel.detach().cpu().item(),
                                  lost_rel_loss     = lost_rel_loss.detach().cpu().item(),
                                  filter_loss_graph = filter_loss_graph.detach().cpu().item(),
                                  prune_loss        = prune_loss.detach().cpu().item(), 
                                  pred_loss_span    = pred_loss_span.detach().cpu().item(),
                                  pred_loss_rel     = pred_loss_rel.detach().cpu().item())

        output = dict(
            loss                 = total_loss,
            loss_breakdown       = loss_breakdown,
            ######################################
            logits_span          = logits_span,
            span_masks           = span_masks,
            span_ids             = span_ids,    
            span_labels          = span_labels,
            ######################################
            logits_rel           = logits_rel,
            rel_masks            = rel_masks,
            rel_ids              = rel_ids,
            rel_labels           = rel_labels,
            lost_rel_counts      = lost_rel_counts,  #how many positive rels did not get into rel_reps, i.e. not missing due to filtering or misclassification
        )

        #print(f'top_k_spans: {top_k_spans}, top_k_rels: {top_k_rels}')

        return output
