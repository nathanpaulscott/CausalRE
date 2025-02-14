import torch
import copy
from torch.profiler import record_function



class Predictor:
    '''
    class to make predictions from the output of the model run    
    '''
    def __init__(self, config):
        self.config = config
        self.all_preds_out = {'spans': [], 'rels': [], 'rels_mod': []}


    def reset_data(self):
        self.all_preds_out = {'spans': [], 'rels': [], 'rels_mod': []}


    def remove_span_types_from_full_rels(self, full_rels):
        '''
        This removes the span types from the full rels as this is required for some analysis
        '''
        return [[(rel[0],rel[1],rel[3],rel[4],rel[6]) for rel in obs] for obs in full_rels]


    def prep_and_add_batch_preds(self, span_preds, rel_preds):
        '''
        function to add the batch of preds (list of list of tuples) to the internal store to be returned later
        '''
        #make the rel_preds without the span types
        rel_preds_mod = self.remove_span_types_from_full_rels(rel_preds)
        #add to the output dicts
        self.all_preds_out['spans'].extend(span_preds)
        self.all_preds_out['rels'].extend(rel_preds)
        self.all_preds_out['rels_mod'].extend(rel_preds_mod)


    def predict_unilabel(self, logits):
        """
        Convert logits to single class predictions by applying softmax
        and then taking the argmax, along with the maximum probability
        for each predicted class.
        This assumes logits are shaped as (batch_size, num_items, num_types).
        Args:
            logits (torch.Tensor): Logits tensor of shape (batch_size, num_items, num_types).
        
        Returns:
            torch.Tensor: Predicted class indices tensor of shape (batch_size, num_items).
            torch.Tensor: Maximum class probabilities tensor of shape (batch_size, num_items).
        """
        probs = torch.softmax(logits, dim=2)  # Apply softmax across the last dimension
        preds = torch.argmax(probs, dim=2)  # Get the index of the max log-probability across types
        max_probs = torch.max(probs, dim=2)[0]  # [0] to select values only
        return preds, max_probs



    def predict_multilabel(self, logits, thd=0.5):
        """
        Convert logits to multilabel predictions by applying sigmoid
        and then using a threshold to determine label assignment.
        This assumes logits are shaped as (batch_size, num_items, num_types).
        Args:
            logits (torch.Tensor): Logits tensor of shape (batch_size, num_items, num_types).
            thd (float): Threshold for determining label assignment.

        Returns:
            torch.Tensor: Predicted labels tensor of shape (batch_size, num_items, num_types),
                          where each element is 0 or 1.
            torch.Tensor: Probabilities tensor of shape (batch_size, num_items, num_types),
                          representing the probability of each type for each item.
        """
        probs = torch.sigmoid(logits)  # Apply sigmoid to convert logits to probabilities
        preds = (probs >= thd).int()  # Apply threshold to determine label assignments
        return preds, probs


    def predict_spans_unilabel(self, model_out):
        '''
        Generates the unilabel preds and probs from teh span_logits
        It then extracts the positive cases from the preds and formats them into the internal list of list of tuples format for comparing to the labels
        '''
        span_logits = model_out['logits_span']
        span_ids    = model_out['cand_w_span_ids']

        #Get predictions and probabilities
        preds, probs = self.predict_unilabel(span_logits)
        
        #convert to list of list of tuples format for evaluation with the labels
        #Initialize the spans output data structure, list of empty lists
        spans = [[] for x in range(span_logits.shape[0])]  # batch_size is the first dimension
        conf = [[] for x in range(span_logits.shape[0])]  # batch_size is the first dimension
        #Get the indices for the postiive cases in preds
        #NOTE: torch.where will return one tensor for each dim in the subject tensor with the idx of each match
        #      thus in this case preds is 2D, so we get 2 tensors for dim 0 (batch indices) and dim 1 (span indices)
        batch_indices, span_indices = torch.where(preds > 0)
        #iterate through the postive pred indices and fill the output data
        for batch_idx, span_idx in zip(batch_indices, span_indices):
            #Extract span details, which are in span_ids tensor at the same idx position
            span_start = span_ids[batch_idx, span_idx, 0].item()
            span_end = span_ids[batch_idx, span_idx, 1].item()
            #extract the pred as an span type idx
            pred_idx = preds[batch_idx, span_idx].item()
            #get the pred as a string
            pred_str = self.config.id_to_s[pred_idx]  # Map class index to label string
            #add to the output
            spans[batch_idx.item()].append((span_start, span_end, pred_str))
            if self.config.predict_conf:   #fill the confidence if requested
                conf[batch_idx.item()].append(probs[batch_idx, span_idx].item())

        return spans, conf, preds


    def predict_spans_multilabel(self, model_out):
        '''
        NOTE: this option has been disabled as it creates too much difficulty in the prediction stage
        NOTE: this option has been disabled as it creates too much difficulty in the prediction stage
        NOTE: this option has been disabled as it creates too much difficulty in the prediction stage
        Extracts the span predictions for multilabel classification in a list of lists of tuples form,
        where each list corresponds to a batch item.
        '''
        span_logits = model_out['logits_span']
        span_ids    = model_out['cand_w_span_ids']

        preds, probs = self.predict_multilabel(span_logits, self.config.predict_thd)  # Get predictions and probabilities
        #Initialize the list to hold batch-wise span information
        spans = [[] for _ in range(span_logits.shape[0])]  # batch_size is the first dimension
        conf = [[] for _ in range(span_logits.shape[0])]  # batch_size is the first dimension
        #Find indices where predictions are positive
        #will return 3 tensors of same length with that dims idx for each pos case 
        batch_indices, span_indices, label_indices = torch.where(preds == 1)
        for batch_idx, span_idx, label_idx in zip(batch_indices, span_indices, label_indices):
            #Extract span details
            span_start = span_ids[batch_idx, span_idx, 0].item()
            span_end = span_ids[batch_idx, span_idx, 1].item()
            label = self.config.id_to_s[label_idx.item()]  # Map label index to label string
            spans[batch_idx.item()].append((span_start, span_end, label))
            if self.config.predict_conf:
                conf[batch_idx.item()].append(probs[batch_idx, span_idx, label_idx].item())

        return spans, conf, preds



    def predict_rels_unilabel(self, model_out, span_preds):
        '''
        Extracts relation predictions for a unilabel classification task from the given logits. 
        This function constructs a list of lists of tuples, where each list corresponds to a batch item.
        
        Each tuple represents a relation and is formatted as follows:
        (head_start, head_end, head_type, tail_start, tail_end, tail_type, rel_type),
        where each element represents the respective span start, end, predicted type of the head and tail,
        and the relation type. A confidence score is optionally included if specified in the configuration.
        
        Parameters:
        - rel_logits (torch.Tensor): Logits for relation types (shape: [batch_size, num_relations, num_relation_types]).
        - rel_ids (torch.Tensor): Indices of head and tail spans (shape: [batch_size, num_relations, 2]).
        - span_ids (torch.Tensor): Start and end indices of spans (shape: [batch_size, num_spans, 2]).
        - span_preds (torch.Tensor): Predicted labels for each span (shape: [batch_size, num_spans]).  NOTE: span preds are unilabel only (one pred per span start,end)
        
        Returns:
        - list of lists of tuples: For each batch item, a list of tuples describing the predicted relations.
        '''
        rel_logits = model_out['logits_rel']
        rel_ids    = model_out['cand_rel_ids']
        span_ids   = model_out['cand_w_span_ids']

        preds, probs = self.predict_unilabel(rel_logits)  # Adjust this method to return predictions and probs for relations
        rels = [[] for _ in range(rel_logits.shape[0])]
        conf = [[] for _ in range(rel_logits.shape[0])]  # batch_size is the first dimension
        #Find indices where predictions are positive (ignoring class 0, the "none" class)
        batch_indices, rel_indices = torch.where(preds > 0)
        for batch_idx, rel_idx in zip(batch_indices, rel_indices):
            #Extract indices for head and tail spans in cand_span_ids
            head_span_idx = rel_ids[batch_idx, rel_idx, 0].item()
            tail_span_idx = rel_ids[batch_idx, rel_idx, 1].item()
            #Get the head span start,end,type for the head and tail spans from cand_span_ids and preds
            head_span_start = span_ids[batch_idx, head_span_idx, 0].item()
            head_span_end = span_ids[batch_idx, head_span_idx, 1].item()
            head_span_type = self.config.id_to_s[span_preds[batch_idx, head_span_idx].item()] 
            #Get the tail span start,end,type for the head and tail spans from cand_span_ids and preds
            tail_span_start = span_ids[batch_idx, tail_span_idx, 0].item()
            tail_span_end = span_ids[batch_idx, tail_span_idx, 1].item()
            tail_span_type = self.config.id_to_s[span_preds[batch_idx, tail_span_idx].item()]
            #Get the relation type prediction
            rel_type = self.config.id_to_r[preds[batch_idx, rel_idx].item()]  # Mapping relation ids to labels
            #make the full rels
            rels[batch_idx.item()].append((head_span_start, head_span_end, head_span_type, tail_span_start, tail_span_end, tail_span_type, rel_type))
            if self.config.predict_conf:
                conf[batch_idx.item()].append(probs[batch_idx, rel_idx].item())

        return rels, conf


    def predict_rels_multilabel(self, model_out, span_preds):
        '''
        Extracts the relation predictions for multilabel classification from the given logits,
        where each relation can have multiple types. This function constructs a list of lists of tuples,
        where each list corresponds to a batch item.

        Each tuple represents a relation and is formatted as follows:
        (head_start, head_end, head_type, tail_start, tail_end, tail_type, [rel_types]),
        where each element represents the respective span start, end, predicted type of the head and tail,
        and a list of predicted relation types. A list of confidence scores for each relation type is optionally included.

        Parameters:
        - rel_logits (torch.Tensor): Logits for relation types (shape: [batch_size, num_relations, num_relation_types]).
        - rel_ids (torch.Tensor): Indices of head and tail spans (shape: [batch_size, num_relations, 2]).
        - span_ids (torch.Tensor): Start and end word token indices of spans (shape: [batch_size, num_spans, 2]).
        - span_preds (torch.Tensor): Predicted labels for each span (shape: [batch_size, num_spans]).   (span_labels are unilabel only)

        Returns:
        - list of lists of tuples: For each batch item, a list of tuples describing the predicted relations.
        '''
        rel_logits = model_out['logits_rel']
        rel_ids    = model_out['cand_rel_ids']
        span_ids   = model_out['cand_w_span_ids']

        preds, probs = self.predict_multilabel(rel_logits, self.config.predict_thd)  # Get predictions and probabilities
        rels = [[] for _ in range(rel_logits.shape[0])]
        conf = [[] for _ in range(rel_logits.shape[0])]  # batch_size is the first dimension
        #Find indices where predictions are positive
        batch_indices, rel_indices, rel_type_indices = torch.where(preds == 1)
        for batch_idx, rel_idx, rel_type_idx in zip(batch_indices, rel_indices, rel_type_indices):
            # Extract indices for head and tail spans
            head_span_idx = rel_ids[batch_idx, rel_idx, 0].item()
            tail_span_idx = rel_ids[batch_idx, rel_idx, 1].item()
            #Get the span start, end, type prediction for the head spans
            head_span_start = span_ids[batch_idx, head_span_idx, 0].item()
            head_span_end = span_ids[batch_idx, head_span_idx, 1].item()
            head_type = self.config.id_to_s[span_preds[batch_idx, head_span_idx].item()]
            #Get the span start, end, type prediction for the tail spans
            tail_span_start = span_ids[batch_idx, tail_span_idx, 0].item()
            tail_span_end = span_ids[batch_idx, tail_span_idx, 1].item()
            tail_type = self.config.id_to_s[span_preds[batch_idx, tail_span_idx].item()]
            #Map relation type index to relation type string
            rel_type = self.config.id_to_r[rel_type_idx.item()]
            #make the full rels
            rels[batch_idx].append((head_span_start, head_span_end, head_type, tail_span_start, tail_span_end, tail_type, rel_type))
            if self.config.predict_conf:
                conf[batch_idx].append(probs[batch_idx, rel_idx, rel_type_idx].item())

        return rels, conf



    def predict_spans(self, model_out):
        """
        Predict spans from model output based on configuration.
        """
        if self.config.span_labels == 'unilabel':
            return self.predict_spans_unilabel(model_out)
        elif self.config.span_labels == 'multilabel':
            raise ValueError('multilabel not supported for spans')
            #return self.predict_spans_multilabel(model_out)


    def predict_rels(self, model_out, span_preds):
        """
        Predict relationships from model output based on configuration.
        """
        if self.config.rel_labels == 'unilabel':
            return self.predict_rels_unilabel(model_out, span_preds)
        elif self.config.rel_labels == 'multilabel':
            return self.predict_rels_multilabel(model_out, span_preds)


    def predict(self, model_out, return_and_reset_results=False):
        """
        Runs predictions for spans and relations based on the model output and configuration.
        Handles both unilabel and multilabel predictions as configured.
        Args:
            model_out (dict): Model output containing logits and candidate ids.
            return_and_reset_results (bool): If True, returns and resets internal prediction storage.

        Returns:
            dict: A deep copy of all predictions if return_and_reset_results is True, otherwise None.
        """
        #get the span labels and preds => slow but 10x faster than the rels
        spans, span_conf, span_preds = self.predict_spans(model_out)
        #get the rel labels and preds => very slow right now
        rels, rel_conf = self.predict_rels(model_out, span_preds)
        #prep and add to the all_preds_out object
        self.prep_and_add_batch_preds(spans, rels)
        #return data and reset if requested
        if return_and_reset_results:
            result = copy.deepcopy(self.all_preds_out)
            self.reset_data()
            return result

