import torch

from .utils import load_from_json, save_to_json
from .validator import Validator





class DataPreparation:
    def __init__(self, main_configs):
        #need the main configs object here as we actually are going to update it
        self.main_configs = main_configs 
        self.config = main_configs.as_namespace
        self.validator = Validator(self.config)


    def make_all_possible_spans(self, max_seq_len, max_span_width):
        '''
        This makes all possible spans indices given:
        - max allowable input word token sequence length (seq_len)
        - max allowable span width in word tokens (self.config.max_span_width)
        
        The output is a list of tuples where each tuple is the start and end token idx for that given span.  
        NOTE: to get the valid spans from a particular seq_len, you can generate a mask with: valid_span_mask = torch.tensor(span_indices, dtype=torch.long)[:, 1] > seq_len
        
        NOTE: the start and end are python list style indices, such that start is the actual word token idx and end is the actual word token idx + 1
        NOTE: remember this calc includes spans that end outside of the sequence length, they are just masked later, this actually makes things easier
              as the num spans here = max_seq_len * max_span_wdith, i.e. each idx in the sequence has max_span_width spans that start on it, a nice useful idea
        '''
        span_indices = []
        for i in range(max_seq_len):
            span_indices.extend([(i, i + j) for j in range(1, max_span_width + 1)])
        return span_indices


    def create_type_mappings(self, types):
        """
        Creates mappings from type names to IDs and vice versa for labeling tasks.

        This function generates dictionaries to map type names to unique integer IDs and vice versa.  
        for unilabels => idx 0 is for the none type (the negative case)
        for multilabels => there is no none type in the types as it is not explicitly given, so idx 0 is for the first pos type

        Parameters:
        - types (list of str): The list of type names for which to create mappings.

        Returns:
        - tuple of (dict, dict): 
            - The first dictionary maps type names to IDs.
            - The second dictionary maps IDs back to type names.
        """
        # Create type to ID mapping
        type_to_id = {t: i for i, t in enumerate(types)}
        # Create ID to type mapping
        id_to_type = {i: t for t, i in type_to_id.items()}

        return type_to_id, id_to_type



    def make_type_mappings_and_span_ids(self):
        """
        Initializes and configures the type mappings and identifier settings for span and relationship types
        within the model's configuration. This method sets several configuration properties related to span
        and relationship types, including creating mapping dictionaries and a list of all possible spans.

        Updates the following in self.config:
        - span_types => adds the none type to the start of the list for unilabels case
        - rel_types => adds the none type to the start of the list for the unilabels case
        - num_span_types: Number of span types   (will be pos and neg types for unilabel and just pos types for multilabel)
        - num_rel_types: Number of relationship types   (will be pos and neg types for unilabel and just pos types for multilabel)
        - s_to_id: Dictionary mapping from span types to their indices.
        - id_to_s: Dictionary mapping from indices to span types.
        - r_to_id: Dictionary mapping from relationship types to their indices.
        - id_to_r: Dictionary mapping from indices to relationship types.
        - all_span_ids: A list of all possible spans generated from the span types.

        No parameters are required as the method operates directly on the class's config attribute.

        NOTE:
        for span/rel_labels == 'unilabels' => span/rel_types will include the none type at idx 0, num_span/rel_types includes the none type
        for span/rel_labels == 'multilabels' => there is no none type in the types as it does not need to be explicitly given, so idx 0 is for the first pos type, num_span/rel_types DOES NOT include the none type
        """
        #add the negative type (none type) for the unilabel case, will take up idx 0
        #add the none type to start of the types lists, it will not be in the list as validation checks would have rejected it
        #NOTE: for the multilabel case the none_span/rel is not required explicitly in types
        config = self.main_configs.as_namespace
        ##############################
        #update the main configs
        self.main_configs.update(dict(
            span_types = ['none'] + config.span_types,
            rel_types  = ['none'] + config.rel_types  if config.rel_labels  == 'unilabel' else config.rel_types
        ))
        ##############################

        #get the num span and rel total types
        #unilabels => will be pos and neg types
        #multilabels => just pos types
        config = self.main_configs.as_namespace    #re-read the current configs
        num_span_types = len(config.span_types)
        num_rel_types  = len(config.rel_types)
        s_to_id, id_to_s = self.create_type_mappings(config.span_types)
        r_to_id, id_to_r = self.create_type_mappings(config.rel_types)
        
        #ake all possible span ids from max_seq_len and max_span_width params, leave as python object for now
        all_span_ids = self.make_all_possible_spans(config.max_seq_len, config.max_span_width)

        #make the lost rel penalty increment tensor
        lost_rel_penalty_incr = torch.tensor(1.0, dtype=config.torch_precision, device=config.device)

        ##############################
        #update the main configs
        self.main_configs.update(dict(
            num_span_types = num_span_types,
            num_rel_types  = num_rel_types,
            s_to_id        = s_to_id,
            id_to_s        = id_to_s,
            r_to_id        = r_to_id,
            id_to_r        = id_to_r,
            all_span_ids   = all_span_ids,
            lost_rel_penalty_incr = lost_rel_penalty_incr
        ))
        ##############################
      


    def convert_id_to_idx_format(self, raw_spans, raw_rels):
        '''
        converts the id format to the idx format
        '''
        #determine the valid span idx which do not violate the imposed limits from config.max_seq_len and config.max_span_width
        id_to_idx_map = {span['id']:i for i,span in enumerate(raw_spans)}
        spans = [{'start': span['start'], 'end': span['end'], 'type': span['type']} for span in raw_spans]
        rels = [{'head':id_to_idx_map[rel['head']], 'tail':id_to_idx_map[rel['tail']], 'type':rel['type']} for rel in raw_rels]

        return spans, rels



    def extract_valid_spans_rels_obs(self, raw_spans, raw_rels):
        '''
        Operates on the raw_spans and raw_rels annotations for one obs
        It filters out the invalid spans, not meeting the requirement to be within 
        the max_seq_len and have span width not exceeding max_span_width
        It then makes a mapping from the raw span idx to the valid span idx and uses this 
        to update the head/tail span idx in rels
        NOTE: the output spans and rels are lists of tuples
        '''
        #convert the id format to list idx format for head/tail
        #I will take this out when I have fixed the js code
        if self.config.data_format == 'id':
            raw_spans, raw_rels = self.convert_id_to_idx_format(raw_spans, raw_rels)

        #determine the valid span idx which do not violate the imposed limits from config.max_seq_len and config.max_span_width
        valid_spans = [
            i for i,span in enumerate(raw_spans) 
            if span['end'] - span['start'] <= self.config.max_span_width and 
            span['start'] < self.config.max_seq_len
        ]
        #make the reverse mapping from valid span idx to raw span idx
        raw2valid = {raw_idx:valid_idx for valid_idx, raw_idx in enumerate(valid_spans)}

        #convert spans and rels annotations to list of tuples from list of dicts and apply the valid span/rel filter
        #NOTE: this will adjust the the rel head/tail idx as the source spans list has changed!!
        spans = [(span['start'], span['end'], span['type']) 
                 for i, span in enumerate(raw_spans) if i in valid_spans]
        
        rels = [(raw2valid[rel['head']], raw2valid[rel['tail']], rel['type']) 
                for rel in raw_rels if rel['head'] in raw2valid and rel['tail'] in raw2valid]

        return spans, rels



    def extract_dataset(self, raw_data):
        '''
        Extracts and structures the dataset based on the run type.
        The spans and relations are also filtered to only include valid spans (within the max_seq_len and width not exceeding max_span_width)
        This filtering will update the relation head/tail references to the span list, so they may be different internally than the incoming annotated data due to this
        NOTE: the spans and relations are converted to list of tuples
        '''
        try:
            dataset = {}
            if self.config.run_type == 'train':
                dataset = {}
                splits = ['train', 'val', 'test']
                for split in splits:
                    data = []
                    for raw_obs in raw_data['data'][split]:
                        #extract the valid spans and rels (ensuring to update the head/tail span idx to the valid idx)
                        valid_spans, valid_rels = self.extract_valid_spans_rels_obs(raw_obs['spans'], raw_obs['relations']) 
                        data.append(dict(tokens    = raw_obs['tokens'], 
                                         spans     = valid_spans, 
                                         relations = valid_rels))
                    dataset[split] = data

            elif self.config.run_type == 'predict':
                dataset = {}
                data = []
                for raw_obs in raw_data['data']['predict']:
                    data.append(dict(tokens = raw_obs['tokens']))
                dataset['predict'] = data
            
            else: 
                raise Exception("invalid run_type, must be 'train' or 'predict'")

            return dataset
        except Exception as e:
            raise ValueError(f'Could not form the dataset from the imported data, there are issues with the format: {e}')




    def import_data(self):
        """
        Load a source dataset from a JSON file and extract its schema.

        Args:
            config => the config namespace
            
        Input JSON Format (train):
            - Outer dictionary with 2 keys: 'data', 'schema'.
                - 'data' is a dict of 3 keys: 'train', 'val', 'test'
                    - 'tokens': List of word tokens for the input text
                    - 'spans': list of dictionaries, where each dictionary represents a span with 3 keys:
                        - 'start': word token index of the span start in the raw text.
                        - 'end': word token index of the span end + 1 in the raw text (actual end token + 1)
                        - 'type': The type of the span.
                    - 'relations': List of dictionaries, where each dict represents a directed relation with 3 keys:
                        - 'head': idx in the spans list for the head span.
                        - 'tail': idx in the spans list for the tail span.
                        - 'type': The type of the relation.
                - 'schema': dict with 2 keys:
                    - 'span_types': List of dictionaries, each defining a span type with:
                        - 'name': The name of the span type.
                        - 'color': The color specification (e.g., rgba(1,2,3,0.3)).
                    - 'relation_types': List of dictionaries, each defining a relation type with:
                        - 'name': The name of the relation type.
                        - 'color': The color specification.

        Input JSON Format (predict):
            - Outer dictionary with 2 keys: 'data', 'schema'.
                - 'data' is a dict of 1 key: 'predict'
                    - 'tokens': List of word tokens for the input text
                - 'schema': dict with 2 keys:
                    - 'span_types': List of dictionaries, each defining a span type with:
                        - 'name': The name of the span type.
                        - 'color': The color specification (e.g., rgba(1,2,3,0.3)).
                    - 'relation_types': List of dictionaries, each defining a relation type with:
                        - 'name': The name of the relation type.
                        - 'color': The color specification.

        Returns:
            tuple: A tuple containing:
                - data (dict): The dataset without the schema key.
                - schema (dict): The extracted schema.

        Raises:
            KeyError: If the 'schema' key is missing from the JSON.
        """
        config = self.config
        #make the absolute data path
        data_path = str(config.app_path / config.data_path)
        #Load the JSON file into a Python object
        result = load_from_json(data_path)
        
        #validate the imported configs
        self.validator.validate_config()
        #do schema validity checks
        self.validator.validate_schema(result)
        #do validation on the import
        self.validator.validate_dataset(result)
       
        #extract only the desired info from the import
        dataset = self.extract_dataset(result)
        #Extract the schema
        schema = result['schema']

        #Extract and sort span and relation types and update the main_configs
        span_types = sorted({x['name'] for x in schema['span_types']})
        rel_types = sorted({x['name'] for x in schema['relation_types']})
        self.main_configs.update({'span_types': span_types, 'rel_types': rel_types})
        ######################################
       
        return dataset




    def load_and_prep_data(self):
        '''
        This imports the data specified in self.config
        It sets additional parameters in self.config
        It validates self.config
        Generates all possible spans and generates associated maps
        Finally, it preprocesses the dataset and returns the data loaders for the next stages
        '''
        self.config.logger.write('Loading and Preparing Data', 'info')

        #load the predict data
        dataset = self.import_data()

        #add to the config => the span and rel type mapping dicts and the all_possible_spans data
        #adds the none_span/rel types to the possible types for the unilabel case (see function for details)
        self.make_type_mappings_and_span_ids()

        return dataset

