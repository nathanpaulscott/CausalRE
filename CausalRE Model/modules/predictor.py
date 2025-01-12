import torch
import copy



class Predictor:
    '''
    class to make predictions from the output of the model run    
    '''
    def __init__(self, config):
        self.config = config
        self.reset_data()


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
        Args:
            logits (torch.Tensor): Logits tensor of shape (batch_size, num_classes).
        
        Returns:
            torch.Tensor: Predicted class indices tensor of shape (batch_size,).
            torch.Tensor: Maximum class probabilities tensor of shape (batch_size,).
        """
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)
        max_probs = torch.max(probs, dim=1)[0]  # [0] to select values only, [1] would give indices which are `preds`
        return preds, max_probs


    def predict_multilabel(self, logits, thd=0.5):
        """
        Convert logits to multilabel predictions by applying sigmoid
        and then using a threshold to determine label assignment.
        Args:
            logits (torch.Tensor): Logits tensor of shape (batch_size, num_labels).
            thd (float): Threshold for determining label assignment.

        Returns:
            torch.Tensor: Predicted labels tensor of shape (batch_size, num_labels),
                        where each element is 0 or 1.
            torch.Tensor: Probabilities tensor of shape (batch_size, num_labels),
                        representing the probability of each label.
        """
        probs = torch.sigmoid(logits)
        preds = (probs >= thd).int()
        return preds, probs



    def predict_spans_unilabel(self, span_logits, span_ids):
        '''
        Extracts the span predictions for unilabel classification in a list of lists of tuples form,
        where each list corresponds to a batch item.
        '''
        preds, probs = self.predict_unilabel(span_logits)  # Get predictions and probabilities
        #Initialize the spans output data structure
        spans = [[] for _ in range(span_logits.shape[0])]  # batch_size is the first dimension
        conf = [[] for _ in range(span_logits.shape[0])]  # batch_size is the first dimension
        #Find indices where predictions are positive (ignoring class 0, the "none" class)
        #will return 2 tensors of same length with that dims idx for each pos case 
        batch_indices, span_indices = torch.where(preds > 0)
        for batch_idx, span_idx in zip(batch_indices, span_indices):
            # Extract span details
            span_start = span_ids[batch_idx, span_idx, 0].item()
            span_end = span_ids[batch_idx, span_idx, 1].item()
            label = self.config.id_to_s[preds[batch_idx, span_idx].item()]  # Map class index to label string
            spans[batch_idx.item()].append((span_start, span_end, label))
            if self.config.predict_conf:
                conf[batch_idx.item()].append(probs[batch_idx, span_idx].item())

        return spans, conf


    def predict_spans_multilabel(self, span_logits, span_ids):
        '''
        NOTE: this option has been disabled as it creates too much difficulty in the prediction stage
        NOTE: this option has been disabled as it creates too much difficulty in the prediction stage
        NOTE: this option has been disabled as it creates too much difficulty in the prediction stage
        Extracts the span predictions for multilabel classification in a list of lists of tuples form,
        where each list corresponds to a batch item.
        '''
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

        return spans, conf



    def predict_rels_unilabel(self, rel_logits, rel_ids, span_ids, span_preds):
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



    def predict_rels_multilabel(self, rel_logits, rel_ids, span_ids, span_preds):
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
        - span_ids (torch.Tensor): Start and end indices of spans (shape: [batch_size, num_spans, 2]).
        - span_preds (torch.Tensor): Predicted labels for each span (shape: [batch_size, num_spans]).   (span_labels are unilabel only)

        Returns:
        - list of lists of tuples: For each batch item, a list of tuples describing the predicted relations.
        '''
        preds, probs = self.predict_multilabel(rel_logits, self.config.predict_thd)  # Get predictions and probabilities
        rels = [[] for _ in range(rel_logits.shape[0])]
        conf = [[] for _ in range(rel_logits.shape[0])]  # batch_size is the first dimension
        # Find indices where predictions are positive
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



    def predict_spans(self, out):
        """
        Predict spans from model output based on configuration.
        """
        if self.config.span_labels == 'unilabel':
            return self.predict_spans_unilabel(out['logits_span'], out['cand_w_span_ids'])
        elif self.config.span_labels == 'multilabel':
            raise ValueError('multilabel not supported for spans')
            #return self.predict_spans_multilabel(out['logits_span'], out['cand_w_span_ids'])


    def predict_rels(self, out):
        """
        Predict relationships from model output based on configuration.
        """
        if self.config.rel_labels == 'unilabel':
            return self.predict_rels_unilabel(out['logits_rel'], out['cand_rel_ids'])
        elif self.config.rel_labels == 'multilabel':
            return self.predict_rels_multilabel(out['logits_rel'], out['cand_rel_ids'])



    def predict(self, out, return_and_reset_results=False):
        """
        Runs predictions for spans and relations based on the model output and configuration.
        Handles both unilabel and multilabel predictions as configured.
        Args:
            out (dict): Model output containing logits and candidate ids.
            return_and_reset_results (bool): If True, returns and resets internal prediction storage.

        Returns:
            dict: A deep copy of all predictions if return_and_reset_results is True, otherwise None.
        """
        spans, span_conf = self.predict_spans(out)
        rels, rel_conf = self.predict_rels(out)
        #prep and add to the all_preds_out object
        self.prep_and_add_batch_preds(spans, rels)
        #return data and reset if requested
        if return_and_reset_results:
            result = copy.deepcopy(self.all_preds_out)
            self.reset_data()
            return result

