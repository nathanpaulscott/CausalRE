from .metrics import Metrics_skl, Metrics_custom



def make_evaluator(config):
    '''
    I have the evaluator like this as I have 2 options, one is using skl functions, that needs the preds/labels to be strings and to be aligned, so it is a bit more of a hassle
    The other is a micro averaging manual way that is faster and simpler, it doesn't need alignment either, nor to conevrt to strings, potentially I will just remove the skl option and 
    simplify this into just one class later
    '''
    if config.eval_type == 'skl':
        return Evaluator_skl(config, Metrics_skl)
    else:
        return Evaluator_custom(config, Metrics_custom)


class EvaluatorBase:
    def __init__(self, config, metrics_class):
        self.config = config
        self.metrics = metrics_class(config)
        self.all_preds = {'spans': [], 'rels': [], 'rels_mod': []}
        self.all_labels = {'tokens': [], 'spans': [], 'rels': [], 'rels_mod': []}

    def prepare_labels(self, span_labels_raw, rel_labels_raw):
            '''
            This extracts the raw annotations for alignment with the preds 
            For rels it makes the full rel format
            NOTE: all this is done in python objects
            span_labels => list of list of tuples [[(span_start, span_end, span_type),...],...]
            rel_labels => list of list of tuples [[(head_span_start, head_span_end, head_span_type, tail_span_start, tail_span_end, tail_span_type, rel_type),...],...]
            '''
            batch = len(span_labels_raw)
            span_labels = [[] for x in range(batch)] 
            rel_labels  = [[] for x in range(batch)]
            rel_labels_mod  = [[] for x in range(batch)]
            #fill the span and rel labels objects
            for batch_idx in range(batch):
                #fill the span_labels
                for start, end, span_type in span_labels_raw[batch_idx]:
                    span_labels[batch_idx].append((start, 
                                                   end, 
                                                   self.config.s_to_id[span_type]))
                #fill the rel_labels
                for head, tail, rel_type in rel_labels_raw[batch_idx]:
                    head_span_start, head_span_end, head_span_type = span_labels[batch_idx][head]
                    tail_span_start, tail_span_end, tail_span_type = span_labels[batch_idx][tail]
                    rel_labels[batch_idx].append((head_span_start, 
                                                  head_span_end, 
                                                  head_span_type, 
                                                  tail_span_start, 
                                                  tail_span_end, 
                                                  tail_span_type, 
                                                  self.config.r_to_id[rel_type]))
                    rel_labels_mod[batch_idx].append((head_span_start, 
                                                      head_span_end, 
                                                      tail_span_start, 
                                                      tail_span_end, 
                                                      self.config.r_to_id[rel_type]))

            return span_labels, rel_labels, rel_labels_mod



    def evaluate(self, return_preds=False, return_labels=False):
        #read in the matching params
        loose_matching = self.config.matching_loose
        tol = self.config.matching_tolerance
        wid = self.config.matching_width_limit
        bin = self.config.matching_make_binary
                
        #Flatten and prepare the data before computing metrics
        #do the preds
        flat_span_preds     = self.flatten_and_prepare(self.all_preds['spans'])
        flat_rel_preds      = self.flatten_and_prepare(self.all_preds['rels'])
        flat_rel_mod_preds  = self.flatten_and_prepare(self.all_preds['rels_mod'])
        #do the labels
        flat_span_labels    = self.flatten_and_prepare(self.all_labels['spans'])
        flat_rel_labels     = self.flatten_and_prepare(self.all_labels['rels'])
        flat_rel_mod_labels = self.flatten_and_prepare(self.all_labels['rels_mod'])
        
        #Compute metrics
        #do the strict metrics
        span_metrics    = self.metrics.run_metrics(flat_span_labels, flat_span_preds)
        rel_metrics     = self.metrics.run_metrics(flat_rel_labels, flat_rel_preds)
        rel_mod_metrics = self.metrics.run_metrics(flat_rel_mod_labels, flat_rel_mod_preds)

        #do the loose metrics
        span_metrics_l, rel_metrics_l, rel_mod_metrics_l = None, None, None
        if loose_matching:
            span_metrics_l    = self.metrics.run_metrics(flat_span_labels, flat_span_preds, loose_matching=loose_matching, tolerance=tol, span_limit=wid, make_binary=bin, type='span')
            rel_metrics_l     = self.metrics.run_metrics(flat_rel_labels, flat_rel_preds, loose_matching=loose_matching, tolerance=tol, span_limit=wid, make_binary=bin, type='rel')
            rel_mod_metrics_l = self.metrics.run_metrics(flat_rel_mod_labels, flat_rel_mod_preds, loose_matching=loose_matching, tolerance=tol, span_limit=wid, make_binary=bin, type='rel_mod')

        return dict(labels                = self.all_labels if return_labels else None,
                    preds                 = self.all_preds if return_preds else None,
                    span_metrics          = span_metrics,
                    rel_metrics           = rel_metrics,
                    rel_mod_metrics       = rel_mod_metrics,
                    span_metrics_l        = span_metrics_l,
                    rel_metrics_l         = rel_metrics_l,
                    rel_mod_metrics_l     = rel_mod_metrics_l)









class Evaluator_custom(EvaluatorBase):
    def __init__(self, config, metrics_class):
        super().__init__(config, metrics_class)

    def prep_and_add_batch(self, preds_and_labels):
        '''
        This takes in the formatted preds (positive cases only as list of list of tuples each for spans and rels) and the raw labels
        It preps them and adds to the output
        NOTE: no need to align for this method
        '''
        #add the preds to the output dicts
        self.all_preds['spans'].extend(preds_and_labels['span_preds'])
        self.all_preds['rels'].extend(preds_and_labels['rel_preds'])
        self.all_preds['rels_mod'].extend(preds_and_labels['rel_mod_preds'])

        #convert the raw labels to the same format as the preds
        #get the labels => the actual positive cases
        #NOTE: the rel_labels are the full rels with the span start,end,type info
        span_labels, rel_labels, rel_labels_mod = self.prepare_labels(preds_and_labels['span_labels'], preds_and_labels['rel_labels'])
        #add the labels to the output dicts
        self.all_labels['spans'].extend(span_labels)
        self.all_labels['rels'].extend(rel_labels)
        self.all_labels['rels_mod'].extend(rel_labels_mod)

        self.all_labels['tokens'].extend(preds_and_labels['tokens'])


    def flatten_and_prepare(self, data):
        '''
        Flattens the pos preds and labels and adds the batch index to each tuple.
        #this nested comprehension is doing the same thing as this....
        output = []
        for i, obs in enumerate(data):
            for item in obs:
                output.append(item + (i,))
        return output
        '''
        return [item + (i,) for i, obs in enumerate(data) for item in obs]























class Evaluator_skl(EvaluatorBase):
    def __init__(self, config, metrics_class):
        super().__init__(config, metrics_class)

    def align_preds_and_labels(self, labels, preds):
        '''
        Aligns the predictions with labels by identifying True Positives, False Negatives, and False Positives.
        Adds placeholders for unmatched labels (FN) and unmatched preds (FP).
        Inputs:
        preds (list of list of tuples): Predicted data for each batch.
        labels (list of list of tuples): True labels for each batch.
        Outputs:
        tuple: A tuple containing two lists (aligned labels and aligned preds).
        '''
        all_labels_aligned = []
        all_preds_aligned = []
        placeholder = ()
        for batch_labels, batch_preds in zip(labels, preds):
            batch_labels = set(batch_labels)
            batch_preds = set(batch_preds)
            TP = batch_labels & batch_preds
            FN = batch_labels - batch_preds
            FP = batch_preds - batch_labels
            #Prepare the output for this batch
            batch_labels_aligned = list(TP) + list(FN) + [placeholder]*len(FP)
            batch_preds_aligned = list(TP) + [placeholder]*len(FN) + list(FP)
            #Append the results to the all_batches lists
            all_labels_aligned.append(batch_labels_aligned)
            all_preds_aligned.append(batch_preds_aligned)

        return all_labels_aligned, all_preds_aligned


    def prep_and_add_batch(self, span_labels_raw, rel_labels_raw, preds):
        '''
        This takes in the formatted preds (positive cases only as list of list of tuples each for spans and rels) and the raw labels
        It preps them and aligns them and adds them to the evaluator output dicts for preds and labels
        '''
        #get the labels => the actual positive cases
        #NOTE: the rel_labels are the full rels with the span start,end,type info
        span_labels, rel_labels, rel_labels_mod = self.prepare_labels(span_labels_raw, rel_labels_raw)
        #align preds and labels
        span_labels, span_preds = self.align_preds_and_labels(span_labels, preds['spans'])
        rel_labels, rel_preds = self.align_preds_and_labels(rel_labels, preds['rels'])
        rel_labels_mod, rel_preds_mod = self.align_preds_and_labels(rel_labels_mod, preds['rels_mod'])

        #add to the output dicts
        self.all_preds['spans'].extend(span_preds)
        self.all_labels['spans'].extend(span_labels)
        self.all_preds['rels'].extend(rel_preds)
        self.all_labels['rels'].extend(rel_labels)
        self.all_preds['rels_mod'].extend(rel_preds_mod)
        self.all_labels['rels_mod'].extend(rel_labels_mod)


    def flatten_and_prepare(self, data):
        '''
        The flattens the preds/labels and stringifies the tuples
        #NOTE: we do not need to add the batch idx as we are using the skl metrics which do not care about repeats
        '''
        #Flatten the list of lists and convert tuples to strings
        return ['_'.join(map(str, item)) for obs in data for item in obs]
        #can add the batch index, but it makes no diff for the skl method, only needed for the manual method with sets
        #return ['_'.join(map(str, item + (i,))) if item != () else '' for i, obs in enumerate(data) for item in obs]










##########################################################################3
#temp
def dump_to_txt(data, path):
    '''
    str(self.config.app_path / self.config.log_folder)
    '''
    # Convert the object to a string representation
    data_str = str(data)

    # Specify the path to your output text file
    file_path = path + '/' + 'debug.txt'

    # Open the text file and write the string representation of the object
    with open(file_path, 'w') as file:
        file.write(data_str)
