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
from .filtering import FilteringLayer
from .layers import MLP, LstmSeq2SeqEncoder, TransLayer, GraphEmbedder
from .loss_functions import compute_matching_loss
from .rel_rep import RelationRep
from .scorer import ScorerLayer
from .span_rep import SpanRepLayer
from .transformer_encoder_flair import TransformerEncoderFlair_w_prompt
from .transformer_encoder_hf import TransformerEncoderHF_w_prompt
from .utils import get_ground_truth_relations, get_candidates, er_decoder, get_relation_with_span, load_from_json, save_to_json
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
            self.transformer_encoder_w_prompt = TransformerEncoderFlair_w_prompt(self.config)
        elif self.config.model_source == 'HF':
            self.transformer_encoder_w_prompt = TransformerEncoderHF_w_prompt(self.config)
        
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


        #this forms the span reps from the token reps using the method defined by config.span_mode,
        #span width embeddings (in word widths)
        self.width_embeddings = nn.Embedding(config.max_span_width, config.width_embedding_size)
        #span representation
        self.span_rep_layer = SpanRepLayer(
            span_mode           = config.span_mode,
            hidden_size         = config.hidden_size,
            max_span_width      = config.max_span_width,    #in word widths
            max_seq_len         = config.max_seq_len,       #in word widths    
            width_embeddings    = self.width_embeddings,    #in word widths
            dropout             = config.dropout,
            ffn_ratio           = config.ffn_ratio, 
            use_span_pos_encoding=config.use_span_pos_encoding,    #whether to use span pos encoding in addition to full seq pos encoding
            pooling             = config.subtoken_pooling,     #whether we are using pooling or not
            cls_flag            = config.model_source == 'HF'    #whether we will have a cls token rep
        )

        # filtering layer for spans and relations
        self._span_filtering = FilteringLayer(config.hidden_size)
        self._rel_filtering = FilteringLayer(config.hidden_size)

        # relation representation
        self.rel_rep_layer = RelationRep(
            config.hidden_size, 
            config.dropout, 
            config.ffn_ratio
        )

        # graph embedder
        #this has code errors
        self.graph_embedder = GraphEmbedder(config.hidden_size)

        # transformer layer
        self.trans_layer = TransLayer(
            config.hidden_size,
            num_heads=config.num_heads,
            num_layers=config.num_transformer_layers
        )

        # keep_mlp
        #this seems to be a simple FFN then a binary classification head
        self.keep_mlp = MLP([config.hidden_size, config.hidden_size * config.ffn_ratio, 1], dropout=0.1)

        # scoring layers
        self.scorer_span = ScorerLayer(
            scoring_type    = config.scorer,
            hidden_size     = config.hidden_size,
            dropout         = config.dropout
        )

        self.scorer_rel = ScorerLayer(
            scoring_type    = config.scorer,
            hidden_size     = config.hidden_size,
            dropout         = config.dropout
        )

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



    def compute_score_train(self, x):
        '''
        this is almost identical to the eval case without the .no_grad
        review if we can merge them!!!
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
        #note that they pass a param in the config file specifying which method to use for span rep generation, he has about 6-7 methods, which is understandable to do ablation studies, 
        #although most of them are weird using convolutions and only 2 very similar ones use the first and last word tokens, what about first + last + maxpooled insides!?!?  What about break into 3rds and max pool each part then concat!?!?
        #what about attention pooling?
        span_reps = self.span_rep_layer(token_reps, 
                                        w_span_ids  = x['span_ids'], 
                                        span_mask   = x['span_mask'], 
                                        sw_span_ids = sw_span_ids, 
                                        cls_reps    = cls_reps,
                                        span_widths = x['span_ids'][:,:,1] - x['span_ids'][:,:,0])

        return dict(span_reps       = span_reps, 
                    span_type_reps  = span_type_reps, 
                    rel_type_reps   = rel_type_reps, 
                    token_reps      = token_reps, 
                    token_masks     = token_masks,
                    sw_span_ids     = sw_span_ids,   #will be None for pooling
                    w2sw_map        = w2sw_map)      #will be None for pooling


    @torch.no_grad()
    def compute_score_eval(self, x, device):
        span_ids = (x['span_ids'] * x['span_mask'].unsqueeze(-1)).to(device)
        # Process input
        result = self.transformer_encoder_w_prompt(x, "eval")
        token_reps     = result['token_reps'] 
        token_masks    = result['token_masks']
        span_type_reps = result['span_type_reps'] 
        rel_type_reps  = result['rel_type_reps']
        cls_reps       = result['cls_reps']         #embeddings for the CLS sw token, only if we are using HF
        sw_span_ids    = result['sw_span_ids']      #tensor (batch, max_seq_len_batch*max_span_width, 2) => x['span_ids'] with values mapped using w2sw_map to the sw token start, end. Only if we are HF with no pooling.
        w2sw_map       = result['w2sw_map']         #w2sw mapping for the non-prompt and non special token word tokens to subword tokens. Only if we are HF with no pooling.

        # Compute representations
        token_reps = self.rnn(token_reps, 
                              token_masks)
        span_reps = self.span_rep_layer(token_reps, 
                                        w_span_ids  = x['span_ids'], 
                                        span_mask   = x['span_mask'], 
                                        sw_span_ids = sw_span_ids, 
                                        cls_reps    = cls_reps,
                                        span_widths = x['span_ids'][:,:,1] - x['span_ids'][:,:,0])

        return dict(span_reps       = span_reps, 
                    span_type_reps  = span_type_reps, 
                    rel_type_reps   = rel_type_reps, 
                    token_reps      = token_reps, 
                    token_masks     = token_masks,
                    sw_span_ids     = sw_span_ids,   #will be None for pooling
                    w2sw_map        = w2sw_map)      #will be None for pooling



    ##################################################################################
    ##################################################################################
    ##################################################################################
    def forward(self, x, prediction_mode=False):
        '''
        x is a batch, which is a dict, with the keys being of various types as described below:
        x['tokens']     => list of ragged lists of strings => the raw word tokenized seq data as strings
        x['spans']      => list of ragged list of tuples => the positive cases for each obs 
        x['relations']  => list of ragged list of tuples => the positive cases for each obs
        x['seq_length'] => tensor (batch) the length of tokens for each obs
        x['span_ids']   => tensor (batch, max_seq_len_batch*max_span_width, 2) => the span_ids truncated to the max_seq_len_batch * max_span_wdith
        x['span_mask']  => tensor (batch, max_seq_len_batch*max_span_width) => 1 for valid spans, 0 for pad and invalid spans
        x['span_label'] => tensor (batch, max_seq_len_batch*max_span_width) => 0 to num_span_types for valid cases, -1 for invalid and pad cases
        '''
        #set some params
        span_label = x['span_label'].clone()
        num_span_types  = len(self.config.span_types)
        num_rel_types   = len(self.config.rel_types)

        # compute span representation
        if prediction_mode:
            # Get the device of the model
            device = next(self.parameters()).device
            #Compute scores for evaluation
            result = self.compute_score_eval(x, device)
        else:
            #Compute scores for training
            result = self.compute_score_train(x)
        #read in the results
        span_reps       = result['span_reps']
        span_type_reps  = result['span_type_reps']
        rel_type_reps   = result['rel_type_reps']
        token_reps      = result['token_reps']
        token_masks     = result['token_masks']

        #check the data here
        #check the data here
        #check the data here
        #check the data here
        #print(x[0])

        print(f'span_reps shape:        {span_reps.shape}')
        print(f'token_reps shape:       {token_reps.shape}')
        print(f'token_masks shape:      {token_masks.shape}')
        print(f'span_type_reps shape:   {span_type_reps.shape}')
        print(f'rel_type_reps shape:    {rel_type_reps.shape}')
        exit()
        #return 0

        '''
        
        we are here now

        I am ready to ove past this point
        '''




        #NEED TO DETERMINE IF THIS SPAN_TYPE_MASK AND REL_TYPE_MASK ARE NEEDED!!!!!!!
        #it is only used here: compute_matching_loss
        #Do not think this type mask shit is correct, need to check it
        #Do nto think this type mask shit is correct, need to check it
        #Do nto think this type mask shit is correct, need to check it
        #Do nto think this type mask shit is correct, need to check it
        # Create masks for relation and entity types, setting all values to 1
        span_type_masks = torch.ones(size=(span_type_reps.shape[0], num_span), device=device)
        rel_type_masks = torch.ones(size=(rel_type_reps.shape[0], num_rel), device=device)

        # Reshape span_rep from (B, L, K, D) to (B, L * K, D)
        #B = batch size, L = seq len, K = max span width, D is the hidden size
        #Nathan: now they reshape the span_rep output back to (batch, num_spans, D)!!  
        #Why the fuck did they not just leave it like that to start with!?
        B, L, K, D = span_reps.shape
        span_reps = span_reps.view(B, L * K, D)

        # Compute filtering scores and CELoss for spans from the span_reps and the span_labels
        #The filter score per span in each obs is a scale from -inf to +inf on the conf of the span being a positive entity (>0) or none_entity (<0)
        #the loss is an accumulated metric over all spans in all obs in the batch, so one scalar for the batch, indicating how well the model can detect that a span is postive case (an entity) or negaitve case (none entity)
        #the binary classification head is trainable, so hopefully it gets better at determining if a span is positive over time
        filter_score_span, filter_loss_span = self._span_filtering(span_reps, x['span_label'])

        # Determine the maximum number of candidates (span candidates - natahn)
        # If L is greater than the configured maximum, use the configured maximum plus an additional top K
        # Otherwise, use L plus an additional top K
        #Nathan => so in the config they have max_top_k = 54 and add_top_k = 10 and max seq len in words at 384
        #so effectively the max span candidates is hard limited to 64 unless L < 54 (short sequence)
        #I know the reason they are limiting this, it is quadratic problem when you try to assess all the possible span-pairs from this, eg 64*2 = 4096 whcih is a lot, ideally you only want 10-20 span candidates
        #They just need to adjust the score thd for filtering the spans to control the max number of spans per seq
        #NOTE this is a problem as below he goes and calcs a per obs top_k, so why not just do that here instead of calculating a scalar
        #the calc should be: max_top_k = tensor cals(min(x['seq_len'], config.max_top_k) + config.add_top_k)
        #then he can apply this tensor here: span_idx_to_keep = sorted_idx[:, :max_top_k], but in a tensor way
        max_top_k = min(L, self.config.max_top_k) + self.config.add_top_k
        # Sort the filter scores for spans in descending order and jsut get the span_idx 
        #NOTE [1] is to get the idx as opposed to the values as torch.sort returns a tuple
        sorted_idx = torch.sort(filter_score_span, dim=-1, descending=True)[1]

        #Basically what he is doing is selecting the top K (usually 64) spans and their associated tensors to use for the intial graph construction
        #but he is doing it in a retarded way!!!
        #I would have just put all the code right here for clarity, we already have B, num_spans = L*K, and D, well easier to read...
        '''
        #this selects the top K spans from each obs
        span_idx_to_keep = sorted_idx[:, :max_top_k]
        candidate_span_reps =    span_reps.gather(1, span_idx_to_keep.unsqueeze(-1).expand(-1, -1, D))
        candidate_span_label =  span_label.gather(1, span_idx_to_keep)
        candidate_span_mask =   x['span_mask'].gather(1, span_idx_to_keep)
        candidate_spans_idx =   x['span_idx'].gather(1, span_idx_to_keep)
        '''
        #this, howvever is the gobbledigook code he has me reading....
        # Define the elements to get candidates for
        #NAthan: make a list of the 4 tensors he wants to filter!!!
        elements = [span_reps, span_label, x['span_mask'], x['span_idx']]
        # Use a list comprehension to get the candidates for each element
        #use a list comp to just fucking make people reading this pull their hair out!
        #and to top it off the 'get_candidates' fn is in another file...
        candidate_span_reps, candidate_span_label, candidate_span_mask, candidate_spans_idx = [
            get_candidates(sorted_idx, element, topk=max_top_k)[0] for element in elements
        ]

        '''
        hang on, the top_k_lengths are the lengths of the sequences in the batch + 10, nothing to do with the candidate span widths
        #See above where he should have put this 
        then they are filtering out candidate spans based on this weird condition where if the cand span is more than seq len + 10 from teh top of the list, then it is removed.
        i.e. say we have 64 candidate spans and the seq len +10 is 45, then he will only keep the top 45, so this only really kicks in for very short sequences
        So basically he is applying an additional candidate pruning strategy based on the length of the seq in words (as opposed to L which is the max seq length)
        This is just half assed stupid shit.  Now the the guy is losing my respect, don't have multiple adhoc pruning steps, it is silly, make your pruning clear and simple
        Why not just apply the previously?!?!?!?!?!?!?!?!?  when you calc max_top_k????  Just calc a tensor, for each obs as oppsoed to a scalar
        Potentially, he first wanted to reduce the tensor sizes for teh tensors for the spans by pruning them on the num_span dim, as you have to keep at a uniform value
        Then he introduced further pruning by masking to individually tailor the num_spans to each obs, maybe, but he could have explained that and done it in a clearer way, that is my issue
        This is key code and it should be super clear.  Also his var names are dogshit.
        '''
        # Calculate the lengths for the top K entities
        #Nathan: so surely they mean they want the span_width for the topK entities in each obs, 
        #but this is not what he is doing, he is just addind 10 to every obs seq length here?!  Nothing to do with the span widths
        top_k_lengths = x["seq_length"].clone() + self.config.add_top_k
        # Create a condition mask where the range of top K is greater than or equal to the top K lengths
        condition_mask = torch.arange(max_top_k, device=span_reps.device).unsqueeze(0) >= top_k_lengths.unsqueeze(-1)
        # Apply the condition mask to the candidate span mask and label, setting the masked values to 0 and -1
        # respectively
        candidate_span_mask.masked_fill_(condition_mask, 0)  #is this even needed as he is putting the masking info in the candidate_span_label tensor!!!
        candidate_span_label.masked_fill_(condition_mask, -1)
        
        #Nathan: now he moves onto relations

        # Get ground truth relations
        #Nathan: he fills the relation_classes tensor with ground truth relation labels from (x['relations]) but reformats it to be aligned with candidate_spans_idx
        #i.e. of shape (batch, max_top_k**2) with all ground truth rels having a relation idx and others having -1
        rel_classes = get_ground_truth_relations(x, candidate_spans_idx, candidate_span_label)
        #rel_rep = self.relation_rep(candidate_span_reps).view(B, max_top_k * max_top_k, -1)  # Reshape in the same line
        #Nathan, they basically just concatenate the span reps for the head and tail spans and reproject the hidden dim back to D
        #Tehy do nto include any context token reps at all!!!!!!  This is notable and could be improved
        rel_reps = self.rel_reps_layer(candidate_span_reps)    #output shape (B, max_top_k, max_top_k, D)
        #move the shape back to 3 dims
        rel_reps = rel_reps.view(B, max_top_k * max_top_k, -1)

        # Compute filtering scores for relations and sort them in descending order
        #Nathan:
        #this is identical code to the span case...
        # Compute filtering scores and CELoss for candidate relations from the candidate_relation_reps and the candidate_relation_labels
        #NOTE: unlike the spans, the rel_reps are already aligned with the candidate spans and so are the rel_labels (relation_classes)
        #The filter score per rel in each obs is a scaled from -inf to +inf on the conf of the rel being a positive rel (>0) or none_rel (<0)
        #the loss is an accumulated metric over all rels in all obs in the batch, so one scalar for the batch, indicating how well the model can detect that a rel is postive case (an relation) or negaitve case (none rel)
        #the binary classification head is trainable, so hopefully it gets better at determining if a rel is positive over time
        #NOTE: the structure of the binary classification head and the score and loss calc is identical to the span case
        filter_score_rel, filter_loss_rel = self._rel_filtering(rel_rep, rel_classes)
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
        elements = [cat_pair_rep.view(B, max_top_k * max_top_k, -1), rel_classes.view(B, max_top_k * max_top_k)]
        # Use a list comprehension to get the candidates for each element
        candidate_pair_rep, candidate_pair_label = [get_candidates(sorted_idx_pair, element, topk=max_top_k)[0] for element in elements]   #NAthan: ill be shape (B, max_top_k**2, D), (B, max_top_k**2)
        # Get the top K relation indices
        topK_rel_idx = sorted_idx_pair[:, :max_top_k]
        # Mask the candidate pair labels using the condition mask and refine the relation representation
        #Nathan: do masking on teh remaining pairs using the same technique as used for the spans
        '''
        Remember this was done in the spans section:
        # Create a condition mask where the range of top K is greater than or equal to the top K lengths
        condition_mask = torch.arange(max_top_k, device=span_reps.device).unsqueeze(0) >= top_k_lengths.unsqueeze(-1)
        '''
        candidate_pair_label.masked_fill_(condition_mask, -1)
        candidate_pair_mask = candidate_pair_label > -1
        ###################################################

        ###################################################
        # Concatenate span and relation representations
        ###################################################
        #Nathan: so he is merging the span_reps and rel_reps into one tensor, he is concat along dim 1, so span reps come first then rel reps, 
        #think of it like a sequence and each span or pair is a token, so he is concat the spans and pairs
        #so shape will be (B, max_top_k + max_top_k*max_top_k, D)
        #he is doing this to throw it into an attention block, the node speific identifiers and the node/edge discriminator shoudl help the attention to dscriminate 
        concat_span_pair = torch.cat((candidate_span_reps, candidate_pair_rep), dim=1)   #Nathan: shape (B, max_top_k + max_top_k**2, D)
        mask_span_pair = torch.cat((candidate_span_mask, candidate_pair_mask), dim=1)   #Nathan: shape (B, max_top_k + max_top_k**2)

        ###################################################
        # Apply transformer layer and keep_mlp
        ###################################################
        #this is a using the torch built in mha transformer encoder, mask is the key padding mask
        #seems to be setup properly, need to check, but looks ok, will be slooooow if you increase layers
        #shape will be same as input (B, max_top_k + max_top_k*max_top_k, D)
        out_trans = self.trans_layer(concat_span_pair, mask_span_pair)    #Nathan: shape will be (B, max_top_k + max_top_k**2, D)
        #the trans out reps go to a FFN then a binary classification head, i.e. last dim goes down to 1, then squeezed out
        #thuse we get one logit per node and edge that we can use to prune the graph
        #Nathan: keep_score will have shape (B, max_top_k + max_top_k*max_top_k)  => (max_top_k nodes + max_top_k^2 edges)
        keep_score = self.keep_mlp(out_trans).squeeze(-1)  # Shape: (B, max_top_k + max_top_k, 1)   #Nathan, this comment is def not correct!! shape will be (B, max_top_k + max_top_k**2, D)

        # Apply sigmoid function and squeeze the last dimension
        # keep_score = torch.sigmoid(keep_score).squeeze(-1)  # Shape: (B, max_top_k + max_top_k)  #Nathan: this comment is also wrong

        # Split keep_score into keep_ent and keep_rel
        #Nathan: this command is wrong and will not work
        #needs to be:
        #keep_span, keep_rel = keep_score.split([max_top_k, max_top_k**2], dim=1)
        keep_span, keep_rel = keep_score.split([max_top_k, max_top_k], dim=1)   #shoudl have out dims (B, max_top_k + max_top_k**2, D)

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
        span_loss = compute_matching_loss(scores_span, candidate_span_label, span_type_masks, num_span)

        # Concatenate labels for binary classification and compute binary classification loss
        span_rel_label = (torch.cat((candidate_span_label, candidate_pair_label), dim=1) > 0).float()
        filter_loss = F.binary_cross_entropy_with_logits(keep_score, ent_rel_label, reduction='none')

        # Compute structure loss and total loss
        structure_loss = (filter_loss * mask_span_pair.float()).sum()
        total_loss = sum([filter_loss_span, filter_loss_rel, rel_loss, span_loss, structure_loss])

        return total_loss
