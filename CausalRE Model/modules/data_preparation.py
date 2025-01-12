from .utils import import_data
from .data_processor import DataProcessor
from .validator import Validator





class DataPreparation:
    def __init__(self, main_configs):
        #need the main configs object here as we actually are going to update it
        self.main_configs = main_configs 



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
        config = self.main_configs.as_namespace()
        #update the main configs
        self.main_configs.update(dict(
            span_types = ['none'] + config.span_types if config.span_labels == 'unilabel' else config.span_types,
            rel_types  = ['none'] + config.rel_types  if config.rel_labels  == 'unilabel' else config.rel_types
        ))

        #get the num span and rel total types
        #unilabels => will be pos and neg types
        #multilabels => just pos types
        config = self.main_configs.as_namespace()    #re-read the current configs
        num_span_types = len(config.span_types),
        num_rel_types  = len(config.rel_types),
        s_to_id, id_to_s = self.create_type_mappings(config.span_types)
        r_to_id, id_to_r = self.create_type_mappings(config.rel_types)
        all_span_ids     = self.make_all_possible_spans(config.max_seq_len, config.max_span_width)
        #update the main configs
        self.main_configs.update(dict(
            num_span_types = num_span_types,
            num_rel_types  = num_rel_types,
            s_to_id        = s_to_id,
            id_to_s        = id_to_s,
            r_to_id        = r_to_id,
            id_to_r        = id_to_r,
            all_span_ids   = all_span_ids
        ))
      


    def load_and_prep_data(self):
        '''
        This imports the data specified in self.config
        It sets additional parameters in self.config
        It validates self.config
        Generates all possible spans and generates associated maps
        Finally, it preprocesses the dataset and returns the data loaders for the next stages
        '''
        #load the predict data
        result = import_data(self.main_configs)
        #get the dataset
        dataset = result['dataset']
        #update the main_configs object
        self.main_configs.update(dict(
            span_types = result['span_types'],
            rel_types  = result['rel_types']
        ))

        #validate the imported configs
        config_validator = Validator(self.main_configs)
        config_validator.validate()

        #add to the config => the span and rel type mapping dicts and the all_possible_spans data
        #adds the none_span/rel types to the possible types for the unilabel case (see function for details)
        self.make_type_mappings_and_span_ids()

        return dataset

        '''
        #make the data loaders
        self.data_processor = DataProcessor(self.main_configs)
        return self.data_processor.create_dataloaders(dataset)
        '''