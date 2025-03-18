import torch
import numpy as np
from torch.profiler import record_function

from .utils import remove_span_types_from_full_rels


class Predictor:
    '''
    class to make predictions from the output of the model run    
    '''
    def __init__(self, config):
        self.config = config
        self.reset_data()


    def reset_data(self):
        self.all_preds_out = {'spans': [], 'rels': [], 'rels_mod': []}


    def prep_and_add_batch_preds(self, span_preds, rel_preds):
        '''
        function to add the batch of preds (list of list of tuples) to the internal store to be returned later
        '''
        #make the rel_preds without the span types
        rel_preds_mod = remove_span_types_from_full_rels(rel_preds)
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
        NOTE: remember for the unilabel case the num_types includes the none type at idx = 0
        So the preds will be positive if the pred id > 0 and neg if the pred id == 0
        Args:
            logits (torch.Tensor): Logits tensor of shape (batch_size, num_items, num_types).
        
        Returns:
            torch.Tensor: Predicted class indices tensor of shape (batch_size, num_items).
            torch.Tensor: Maximum class probabilities tensor of shape (batch_size, num_items).
        """
        probs = torch.softmax(logits, dim=2)  # Apply softmax across the last dimension
        preds = torch.argmax(probs, dim=2)  # Get the index of the max log-probability across types
        max_probs = torch.max(probs, dim=2)[0] if self.config.predict_conf else None
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
        Generates the unilabel preds and probs from the span_logits
        It then extracts the positive cases from the preds and formats them into the internal list of list of tuples format for comparing to the labels
        '''
        span_logits = model_out['logits_span']
        span_ids    = model_out['span_ids']
        span_masks  = model_out['span_masks']

        #Get predictions and probabilities
        preds, probs = self.predict_unilabel(span_logits)
        #Apply span masks to the predictions (consider only positive predictions that are also unmasked)
        valid_preds = (preds > 0) & span_masks
        #Get the indices for the valid positive cases in preds as we only need these for the eval
        batch_indices, span_indices = torch.where(valid_preds)

        #Convert tensor data to NumPy arrays for efficient access
        all_span_starts = span_ids[batch_indices, span_indices, 0].cpu().numpy()
        all_span_ends = span_ids[batch_indices, span_indices, 1].cpu().numpy()
        all_span_types = preds[batch_indices, span_indices].cpu().numpy()
        batch_indices = batch_indices.cpu().numpy()

        # Initialize the spans output data structure, list of empty lists
        spans = [[] for _ in range(span_logits.shape[0])]  # batch_size is the first dimension
        conf = [[] for _ in range(span_logits.shape[0])]  # if confidence data is needed
        #Iterate through the positive pred indices and fill the output data
        for i in range(len(batch_indices)):
            spans[batch_indices[i]].append((all_span_starts[i], 
                                            all_span_ends[i], 
                                            all_span_types[i]))

            # Fill the confidence if requested
        if self.config.predict_conf:
        #    conf[batch_indices[i]].append(probs[batch_indices[i], span_indices[i]].item())  # Assuming conf still needs item()
            pass

        return spans, conf, preds



    def predict_rels_unilabel(self, model_out, span_preds):
        '''
        Extracts positive relation predictions for a unilabel classification task from the given logits. 
        This function constructs a list of lists of tuples, where each list corresponds to a batch item.
        
        Each tuple represents a relation and is formatted as follows:
        (head_start, head_end, head_type, tail_start, tail_end, tail_type, rel_type),
        where each element represents the respective span start, end, predicted type of the head and tail,
        and the relation type. A confidence score is optionally included if specified in the configuration.
        
        Parameters:
        - rel_logits (torch.Tensor): Logits for relation types (shape: [batch_size, num_relations, num_relation_types]).
        - rel_ids (torch.Tensor): Indices of head and tail spans (shape: [batch_size, num_relations, 2]).
        - span_ids (torch.Tensor): Start and end indices of spans (shape: [batch_size, num_spans, 2]).
        - span_preds (torch.Tensor): Predicted labels for each span (shape: [batch_size, num_spans]).
        
        Returns:
        - list of lists of tuples: For each batch item, a list of tuples describing the predicted relations.
        '''
        rel_logits = model_out['logits_rel']
        rel_ids    = model_out['rel_ids']
        span_ids   = model_out['span_ids']
        rel_masks  = model_out['rel_masks']

        #Generate predictions and extract them where they are positive
        preds, probs = self.predict_unilabel(rel_logits)  # Adjust this method to return predictions and probs for relations
        #Apply rel masks to the predictions (consider only positive predictions that are also unmasked)
        valid_preds = (preds > 0) & rel_masks
        #Get the indices for the valid positive cases in preds as we only need these for the eval
        batch_indices, rel_indices = torch.where(valid_preds)

        #Collect tensor data and move to CPU, then convert to numpy for faster processing
        all_head_span_ids = rel_ids[batch_indices, rel_indices, 0]
        all_tail_span_ids = rel_ids[batch_indices, rel_indices, 1]
        all_head_spans = span_ids[batch_indices, all_head_span_ids].cpu().numpy()
        all_tail_spans = span_ids[batch_indices, all_tail_span_ids].cpu().numpy()
        all_head_types = span_preds[batch_indices, all_head_span_ids].cpu().numpy()
        all_tail_types = span_preds[batch_indices, all_tail_span_ids].cpu().numpy()
        all_rel_types = preds[batch_indices, rel_indices].cpu().numpy()
        batch_indices = batch_indices.cpu().numpy()

        # Initialize list of lists for output
        rels = [[] for _ in range(rel_logits.shape[0])]
        conf = [[] for _ in range(rel_logits.shape[0])]  # if confidence data is needed
        for i in range(len(batch_indices)):
            rels[batch_indices[i]].append((all_head_spans[i][0], 
                                           all_head_spans[i][1], 
                                           all_head_types[i],
                                           all_tail_spans[i][0], 
                                           all_tail_spans[i][1], 
                                           all_tail_types[i],
                                           all_rel_types[i]))

        # Fill the confidence if requested
        if self.config.predict_conf:
        #    conf[batch_indices[i]].append(probs[batch_indices[i], span_indices[i]].item())  # Assuming conf still needs item()
            pass
            
        return rels, conf


    def predict_rels_multilabel(self, model_out, span_preds):
        '''
        Extracts the pos relation predictions for multilabel classification from the given logits,
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
        rel_ids = model_out['rel_ids']
        span_ids = model_out['span_ids']
        rel_masks = model_out['rel_masks']

        #get the preds
        preds, probs = self.predict_multilabel(rel_logits, self.config.predict_thd)  # Get predictions and probabilities
        #Apply rel masks to the predictions (consider only positive predictions that are also unmasked)
        valid_preds = (preds > 0) & rel_masks.unsqueeze(-1)
        #Get the indices for the valid positive cases in preds as we only need these for the eval
        batch_indices, rel_indices, rel_type_indices = torch.where(valid_preds)

        #Gather span data an dmove to numpy for efficient scalar access
        all_head_span_ids = rel_ids[batch_indices, rel_indices, 0]
        all_tail_span_ids = rel_ids[batch_indices, rel_indices, 1]
        all_head_spans = span_ids[batch_indices, all_head_span_ids].cpu().numpy()
        all_tail_spans = span_ids[batch_indices, all_tail_span_ids].cpu().numpy()
        all_head_types = span_preds[batch_indices, all_head_span_ids].cpu().numpy()
        all_tail_types = span_preds[batch_indices, all_tail_span_ids].cpu().numpy()
        all_rel_types = rel_type_indices.cpu().numpy()
        batch_indices = batch_indices.cpu().numpy()

        # Initialize list of lists for output
        rels = [[] for _ in range(rel_logits.shape[0])]
        conf = [[] for _ in range(rel_logits.shape[0])]  # if confidence data is needed
        for i in range(len(batch_indices)):
            rels[batch_indices[i]].append((all_head_spans[i][0], 
                                           all_head_spans[i][1], 
                                           all_head_types[i],
                                           all_tail_spans[i][0], 
                                           all_tail_spans[i][1], 
                                           all_tail_types[i],
                                           all_rel_types[i]))

        # Fill the confidence if requested
        if self.config.predict_conf:
        #    conf[batch_indices[i]].append(probs[batch_indices[i], rel_indices[i]].item())  # Assuming conf still needs item()
            pass

        return rels, conf




    def predict_spans(self, model_out):
        """
        Predict spans from model output based on configuration.
        """
        return self.predict_spans_unilabel(model_out)


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
        #get the span labels and preds
        spans, span_conf, span_preds = self.predict_spans(model_out)
        #get the rel labels and preds
        rels, rel_conf = self.predict_rels(model_out, span_preds)
        #prep and add to the all_preds_out object
        self.prep_and_add_batch_preds(spans, rels)
        #return data and reset if requested
        if return_and_reset_results:
            result = {}
            for key, batches in self.all_preds_out.items():
                result[key] = [list(batch) for batch in batches]
            #now reset the original data
            self.reset_data()
            return result




