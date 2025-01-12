import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import copy


class Evaluator:
    def __init__(self, config):
        self.config = config
        self.all_preds = {'spans': [], 'rels': [], 'rels_mod': []}
        self.all_labels = {'spans': [], 'rels': [], 'rels_mod': []}



    def prepare_labels(self, span_labels_raw, rel_labels_raw):
        '''
        This extracts the raw annotations for alignment with the preds 
        For rels it makes the full rel format
        NOTE: all this is done in python objects
        span_labels => list of list of tuples [[(span_start, span_end, span_type),...],...]
        rel_labels => list of list of tuples [[(head_span_start, head_span_end, head_span_type, tail_span_start, tail_span_end, tail_span_type, rel_type),...],...]
        '''
        batch = len(span_labels_raw)
        span_labels = [[] for x in batch] 
        rel_labels = [[] for x in batch]
        #fill the span and rel labels
        for batch_idx in range(batch):
            for start, end, span_type in span_labels_raw[batch_idx]:
                span_labels[batch_idx].append((start, end, span_type))

            for head, tail, rel_type in rel_labels_raw[batch_idx]:
                head_span_start, head_span_end, head_span_type = span_labels[batch_idx][head]
                tail_span_start, tail_span_end, tail_span_type = span_labels[batch_idx][tail]
                rel_labels[batch_idx].append((head_span_start, head_span_end, head_span_type, tail_span_start, tail_span_end, tail_span_type, rel_type))

        return span_labels, rel_labels



    def remove_span_types_from_full_rels(self, full_rels):
        '''
        This removes the span types from the full rels as this is required for some analysis
        '''
        return [[(rel[0],rel[1],rel[3],rel[4],rel[6]) for rel in obs] for obs in full_rels]



    def align_preds_and_labels(self, preds, labels, tuple_size):
        '''
        this aligns the preds to the labels and adds any preds without a match to the end of the list and puts a placeholder in labels.
        Any label with no pred is given a placeholder in preds
        the preds and labels need to be python objects, list of list of tuples
        '''
        # Initialize the aligned lists to hold all batches
        all_labels_aligned = []
        all_preds_aligned = []
        batch = len(labels)
        for batch_idx in range(batch):
            #Create sets for quick lookup
            labels_set = set(labels[batch_idx])
            preds_set = set(preds[batch_idx])
            #Create the default tuple of None values
            null_tuple = tuple([None] * tuple_size)
            #Initialize the combined list and a list to hold unmatched tuples from preds
            labels_preds_combined, unmatched_preds = [], []
            #Match each tuple in labels with preds
            for label_tuple in labels[batch_idx]:
                match_tuple = label_tuple if label_tuple in preds_set else null_tuple
                labels_preds_combined.append((label_tuple, match_tuple))
            #Check for tuples in preds not in labels, sort them, and append to the list
            for pred_tuple in sorted(preds_set - labels_set):
                unmatched_preds.append((null_tuple, pred_tuple))  # Add with labels part as default tuple
            #Append unmatched and sorted tuples from preds to the main list
            labels_preds_combined.extend(unmatched_preds)
            #Splitting labels_preds_combined into labels_aligned and preds_aligned
            labels_aligned = [label for label, pred in labels_preds_combined]
            preds_aligned = [pred for label, pred in labels_preds_combined]
        
            #Append current batch alignment to the all batches aligned lists
            all_labels_aligned.append(labels_aligned)
            all_preds_aligned.append(preds_aligned)

        return all_labels_aligned, all_preds_aligned



    def flatten_and_stringify(self, data):
        '''
        The flattens the preds/labels and stringifies the tuples
        '''
        # Flatten the list of lists and convert tuples to strings
        output = []
        for obs in data:
            # Use a list comprehension to convert each tuple in the sublist to a string
            output.extend('_'.join(map(str, item)) for item in obs)
        return output


    def run_metrics(self, preds, labels):
        '''
        this gets the current overall metrics for use during training

        inputs:
            - aligned, cpu'd, numpy'd, flattened, filtered preds list
            - aligned, cpu'd, numpy'd, flattened, filtered labels list
            - params object

        outputs:
            - metrics dict

        '''
        if len(labels) == 0:
            accuracy, precision, recall, f1 = 0,0,0,0
        else:
            # Calculate overall  metrics
            accuracy = accuracy_score(labels, preds)
            precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average=self.config.f1_ave, zero_division=0)

        metrics = dict(support   = len(labels),
                       accuracy  = accuracy,
                       precision = precision,
                       recall    = recall,
                       f1        = f1)
        metrics['msg'] = f"S: {metrics['support']:.2%}\tP: {metrics['precision']:.2%}\tR: {metrics['recall']:.2%}\tF1: {metrics['f1']:.2%}\n"
        
        return metrics




    def prep_and_add_batch(self, preds, span_labels_raw, rel_labels_raw):
        #get the labels => the actual positive cases
        #NOTE: the rel_labels are the full rels with the span start,end,type info
        span_labels, rel_labels = self.prepare_labels(span_labels_raw, rel_labels_raw)
        #make the rel_labels and preds without the span types
        #rel_preds_mod = self.remove_span_types_from_full_rels(rel_preds)
        rel_labels_mod = self.remove_span_types_from_full_rels(rel_labels)
        #align and flatten
        span_preds, span_labels = self.align_preds_and_labels(preds['spans'], span_labels, 3)
        rel_preds, rel_labels = self.align_preds_and_labels(preds['rels'], rel_labels, 7)
        rel_preds_mod, rel_labels_mod = self.align_preds_and_labels(preds['rels_mod'], rel_labels_mod, 5)
        #add to the output dicts
        self.all_preds['spans'].extend(span_preds)
        self.all_labels['spans'].extend(span_labels)
        self.all_preds['rels'].extend(rel_preds)
        self.all_labels['rels'].extend(rel_labels)
        self.all_preds['rels_mod'].extend(rel_preds_mod)
        self.all_labels['rels_mod'].extend(rel_labels_mod)

    

    def evaluate(self, return_preds=False):
        # Flatten and stringify the data before computing metrics
        flat_span_preds     = self.flatten_and_stringify(self.all_preds['spans'])
        flat_span_labels    = self.flatten_and_stringify(self.all_labels['spans'])
        flat_rel_preds      = self.flatten_and_stringify(self.all_preds['rels'])
        flat_rel_labels     = self.flatten_and_stringify(self.all_labels['rels'])
        flat_rel_mod_preds  = self.flatten_and_stringify(self.all_preds['rels_mod'])
        flat_rel_mod_labels = self.flatten_and_stringify(self.all_labels['rels_mod'])
        # Compute metrics
        span_metrics    = self.run_metrics(flat_span_preds, flat_span_labels)
        rel_metrics     = self.run_metrics(flat_rel_preds, flat_rel_labels)
        rel_mod_metrics = self.run_metrics(flat_rel_mod_preds, flat_rel_mod_labels)

        return dict(span_metrics    = span_metrics,
                    rel_metrics     = rel_metrics,
                    rel_mod_metrics = rel_mod_metrics,
                    preds           = self.all_preds if return_preds else None)
