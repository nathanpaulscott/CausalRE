import logging, json, statistics

from .utils import load_from_json, save_to_json



class Validator:
    '''
    class to validate the incoming config data
    '''
    def __init__(self, config):
        self.config = config


    def validate_config(self):
        '''
        this validates the contents of config
        '''
        #check that the span_labels and rel_labels value is valid
        #DESCRIPTION ON WHY i AM ONLY ALLOWING SPANS TO BE UNILABEL
        #I changed it to only allow span_labels to be unilabel as it makes the final rel predictions to complicated to extract if spans are multilabel
        #and spans never need to be multilabel.  The issue is that the annotation format encodes the span type and rel type for each rel in a convoluted way
        #this is NOT maintained in the data that comes out of the model, nor the model internal workings, i.e. the predicted rel type gives us the head and tail cand_span_ids, but this only 
        #allows us to get the head/tail start,end positions, but for a multilabel case it allows us to get the binary vector of head/tail span type preds, not
        #just one pred for each as in the unilabel case, thus we could have more than 1 type for the head or tail span, thus the way data is stored in the model
        #allows this uncertainty.  To avoid this uncertainty we would actually have to encode each predicted span type into the rel_reps etc and that is NOT what we are doing
        #it woudl be very impractical to do it also as we woudl expand the rel_reps even more for the quadratic of num_head_types * num_tail_types * top_k_spans**2!!!!.  So do not even think about it. 
        #The most practical solution is to just fix the span labels to unilabel so that each span can only have one label, this solves that complexity problem and it reflects the reality also.
        if self.config.span_labels != 'unilabel' or self.config.rel_labels not in ['unilabel', 'multilabel']:
        #if config.span_labels not in ['unilabel', 'multilabel'] or config.rel_labels not in ['unilabel', 'multilabel']:
            raise ValueError('the span_labels must be "unilabel" and rel_labels can only be "unilabel" or "multilabel", exiting.....')




    def validate_dataset(self, raw_data):
        '''
        this validates the contents of the dataset
        '''
        #do validity checks for data
        ######################################
        splits = ['train', 'val', 'test']
        keys = ['tokens', 'spans', 'relations']
        span_keys = ['start', 'end', 'type']
        rel_keys = ['head', 'tail', 'type']
        if self.config.run_type == 'predict':
            splits = ['predict']
            keys = ['tokens']
        try:
            if raw_data is None or 'data' not in raw_data: raise Exception
            for split in splits:
                if split not in raw_data['data']: raise Exception
                for item in raw_data['data'][split]:
                    for key in keys:
                        if key not in item: raise Exception
                    for span in item['spans']:
                        for key in span_keys:
                            if key not in span: raise Exception
                    for rel in item['relations']:
                        for key in rel_keys:
                            if key not in rel: raise Exception
        except Exception as e:
            raise ValueError(f"Invalid data provided. Ensure it contains these splits: '{','.join(splits)}' and each item in each split contains these keys: '{','.join(keys)}' and each span contains 'start','end','type' and each rel contains 'head','tail','type'.")

        #get seq_len stats
        seq_lens = []
        for split in splits:
            for i_obs, obs in enumerate(raw_data['data'][split]):
                seq_lens.append(len(obs['tokens']))

        #check how many spans (and related rels) are missed due to the max_span_width or max_seq_len limits
        missed_spans = {}
        span_cnt = 0
        span_widths = []
        for split in splits:
            for i_obs, obs in enumerate(raw_data['data'][split]):
                for i_span, span in enumerate(obs['spans']):
                    span_cnt += 1
                    fail = []
                    start, end = span['start'], span['end']
                    span_widths.append(span['end'] - span['start'])
                    if start >= self.config.max_seq_len: 
                        fail.append('starts after max_seq_len limit')
                    if end - start > self.config.max_span_width: 
                        fail.append('span width more than max_span_width limit')
                    if len(fail) > 0:
                        missed_spans[f'{split}_obs{i_obs}_span{i_span}_{start}_{end}'] = fail
        missed_cnt = len(missed_spans.keys())
        
        #report stats on the span widths and seq_lens
        msg = '\nseq_len and span_width stats, use these to tune max_seq_len and max_span_widths....\n-------------------------------'
        msg += f'\nseq_len stats (max, mean, sd): {max(seq_lens)}, {round(sum(seq_lens)/len(seq_lens),2)}, {round(statistics.stdev(seq_lens),2)}'
        msg += f'\nspan_width stats (max, mean, sd): {max(span_widths)}, {round(sum(span_widths)/len(span_widths),2)}, {round(statistics.stdev(span_widths),2)}'
        msg += '\n-------------------------------'
        self.config.logger.write(msg, 'info')

        if missed_cnt > 0:
            msg = f'WARNING!! Can not import {round(100*missed_cnt/span_cnt,2)}% of the annotated spans ({missed_cnt} spans) due to max_seq_len and max_span_lenght limits.\nThese spans and associated relations will be ignored, you should assess whether this is a significant issue or not.'
            msg += '\n-------------------------------'
            self.config.logger.write(msg, 'warning')
            if self.config.dump_missed_spans_on_import:
                msg = ''
                for item in missed_spans:
                    msg += '\n' + json.dumps(item)
                self.config.logger.write(msg, 'info')
            #save to json
            #save_to_json(missed_spans, str(self.config.app_path / self.config.log_folder) + '\\missed_spans.json')



    def validate_schema(self, raw_data):
        '''
        this validates the contents of the dataset schema
        '''
       
        if 'schema' not in raw_data:
            raise KeyError("The provided JSON file does not contain the required 'schema' key.")
        if 'span_types' not in raw_data['schema'] or 'relation_types' not in raw_data['schema']:
            raise ValueError("Invalid schema provided. Ensure it contains 'span_types' and 'relation_types'.")

        #validate the span and rel types in the schema
        span_types = sorted({x['name'] for x in raw_data['schema']['span_types']})
        rel_types = sorted({x['name'] for x in raw_data['schema']['relation_types']})
        #check that there are span_Types adn rel_types
        if ((span_types is None or not isinstance(span_types, list) or len(span_types) == 0) or
            (rel_types is None or not isinstance(rel_types, list) or len(rel_types) == 0)):
            raise ValueError('TTER needs to ba one or more span_types and rel_types in the schema and it should match the annotated data types, exiting.....')
        #check that none_span and none_rel do not exist as a type in the schema
        if 'none' in span_types or 'none' in rel_types:
            raise ValueError('none is a reserved type for the negative classification, it should not be in the schema, exiting.....')
        
       