import torch
import json, random, os
import numpy as np
from pathlib import Path




def set_all_seeds(seed=42):
    random.seed(seed)                # Python random module
    np.random.seed(seed)             # Numpy module
    torch.manual_seed(seed)          # PyTorch random number generator
    torch.cuda.manual_seed(seed)     # CUDA random number generator if using GPU
    torch.cuda.manual_seed_all(seed) # CUDA random number generator for all GPUs
    torch.backends.cudnn.deterministic = True  # Makes CUDA operations deterministic
    torch.backends.cudnn.benchmark = False     # Disables CUDA convolution benchmarking for reproducibility
    os.environ['PYTHONHASHSEED'] = str(seed)  # Sets Python hash seed




def decode_relations(id_to_rel, logits, pair_indices, max_pairs, span_indices, threshold=0.5, output_confidence=False):
    # Apply sigmoid function to logits
    probabilities = torch.sigmoid(logits)

    # Initialize list of relations
    relations = [[] for _ in range(len(logits))]

    # Get indices where probability is greater than threshold
    above_threshold_indices = (probabilities > threshold).nonzero(as_tuple=True)

    # Iterate over indices where probability is greater than threshold
    for batch_idx, position, class_idx in zip(*above_threshold_indices):
        # Get relation label
        label = id_to_rel[class_idx.item() + 1]

        # Get predicted pair index
        predicted_pair_idx = pair_indices[batch_idx, position].item()

        # Unravel predicted pair index into head and tail
        head_idx, tail_idx = np.unravel_index(predicted_pair_idx, (max_pairs, max_pairs))

        # Convert head and tail indices to tuples
        head = tuple(span_indices[batch_idx, head_idx].tolist())
        tail = tuple(span_indices[batch_idx, tail_idx].tolist())

        # Get confidence
        confidence = probabilities[batch_idx, position, class_idx].item()

        # Append relation to list
        if output_confidence:
            relations[batch_idx.item()].append((head, tail, label, confidence))
        else:
            relations[batch_idx.item()].append((head, tail, label))

    return relations


def decode_entities(id_to_ent, logits, span_indices, threshold=0.5, output_confidence=False):
    # Apply sigmoid function to logits
    probabilities = torch.sigmoid(logits)

    # Initialize list of entities
    entities = []

    # Get indices where probability is greater than threshold
    above_threshold_indices = (probabilities > threshold).nonzero(as_tuple=True)

    # Iterate over indices where probability is greater than threshold
    for batch_idx, position, class_idx in zip(*above_threshold_indices):
        # Get entity label
        label = id_to_ent[class_idx.item() + 1]

        # Get confidence
        confidence = probabilities[batch_idx, position, class_idx].item()

        # Append entity to list
        if output_confidence:
            entities.append((tuple(span_indices[batch_idx, position].tolist()), label, confidence))
        else:
            entities.append((tuple(span_indices[batch_idx, position].tolist()), label))

    return entities


def er_decoder(x, entity_logits, rel_logits, topk_pair_idx, max_top_k, 
               candidate_spans_idx, threshold=0.5, output_confidence=False):
    '''
    describe this function
    '''
    entities = decode_entities(x["id_to_ent"], 
                               entity_logits, 
                               candidate_spans_idx, 
                               threshold, 
                               output_confidence)
    relations = decode_relations(x["id_to_rel"], 
                                 rel_logits, 
                                 topk_pair_idx, 
                                 max_top_k, 
                                 candidate_spans_idx, 
                                 threshold,
                                 output_confidence)
    return entities, relations


def get_relation_with_span(x):
    '''
    not sure what this is doing, used for eval
    will get to it later
    '''
    spans, relations = x['spans'], x['relations']
    B = len(spans)
    relation_with_span = [[] for i in range(B)]
    for i in range(B):
        rel_i = relations[i]
        span_i = spans[i]
        for rel in rel_i:
            act = (span_i[rel[0]], span_i[rel[1]], rel[2])
            relation_with_span[i].append(act)
    return relation_with_span



##################################################################################
#Generic functions
##################################################################################
##################################################################################
def check_utf_encoding(file_path):
    with open(file_path, 'rb') as file:
        first_three_bytes = file.read(3)
        if first_three_bytes == b'\xef\xbb\xbf':
            print(f"The file {file_path} is encoded with UTF-8-SIG.")
            return 'utf-8-sig'
        else:
            print(f"The file {file_path} is encoded with UTF-8 (no BOM).")
            return 'utf-8'




def load_from_json(filename, encoding=None):
    """
    Load data from a JSON file.
    Args:
        filename (str): The path to the JSON file to be loaded.
    Returns:
        dict: The data loaded from the JSON file, or None if an error occurs.
    Raises:
        FileNotFoundError: If the JSON file does not exist.
        json.JSONDecodeError: If the file is not a valid JSON.
    """
    #check if it is 'utf-8-sig'
    encoding = check_utf_encoding(filename)

    try:
        with open(filename, 'r', encoding=encoding) as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        print(f"Error: The file {filename} does not exist.")
        raise
    except UnicodeDecodeError:
        print(f"Error: The file {filename} cannot be decoded with {encoding} encoding.")
        raise
    except json.JSONDecodeError:
        print(f"Error: The file {filename} contains invalid JSON.")
        raise
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        raise




def save_to_json(data, filename):
    """
    Save data to a JSON file.
    Args:
        data (dict): The data to save.
        filename (str): The path where the JSON file will be saved.
    Raises:
        TypeError: If the data provided is not serializable.
        IOError: If there are issues writing to the file.
    """
    try:
        with open(filename, 'w') as f:
            json.dump(data, f, indent=4)
    except TypeError as e:
        print(f"Error: The data provided contains non-serializable types.")
        raise
    except IOError as e:
        print(f"Error: Unable to write to the file {filename}.")
        raise
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        raise



def import_data(data_path, config):
    """
    Load a source dataset from a JSON file and extract its schema.

    Args:
        config => the config namespace
        
    Input JSON Format (train):
        - Outer dictionary with 2 keys: 'data', 'schema'.
            - 'data' is a dict of 3 keys: 'train', 'val', 'test'
                - each key contains a list of dicts, each dict with 3 keys:
                    - 'tokens': List of word tokens for the input text
                    - 'spans': list of dictionaries, where each dictionary represents a span with 4 keys:
                        - 'id': Span ID (format: E_obs_idx_span_idx).
                        - 'start': Character index of the span start in the raw text.
                        - 'end': Character index of the span end in the raw text (not inclusive, true end index).
                        - 'type': The type of the span.
                    - 'relations': List of dictionaries, where each dict represents a directed relation with 4 keys:
                        - 'id': Relation ID.
                        - 'head': Span ID of the head entity.
                        - 'tail': Span ID of the tail entity.
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
                - the only key contains a list of dicts, each dict has one key:
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
    #make the absolute data path
    data_path = str(config.app_path / config.data_path)
    #Load the JSON file into a Python object
    result = load_from_json(data_path)
    
    #do validity checks for data
    ######################################
    splits = ['train', 'val', 'test']
    keys = ['tokens', 'spans', 'relations']
    if config.run_type == 'predict':
        splits = ['predict']
        keys = ['tokens']
    try:
        if result is None or 'data' not in result: raise Exception
        splits = ['train', 'val', 'test']
        keys = ['tokens', 'spans', 'relations']
        for split in splits:
            if split not in result['data']: raise Exception
            for item in result['data'][split]:
                for key in keys:
                    if key not in item: raise Exception

    except Exception as e:
        raise ValueError(f"Invalid data provided. Ensure it contains these splits: '{','.join(splits)}' and each item in each split contains these keys: '{','.join(keys)}'")

    #ensure that we only select the desired data in the dataset
    dataset = {}
    if config.run_type == 'train':
        dataset = dict(
            train   = [dict(tokens    = x['tokens'],
                            spans     = x['spans'],
                            relations = x['relations'])
                            for x in result['data']['train']],
            val     = [dict(tokens    = x['tokens'],
                            spans     = x['spans'],
                            relations = x['relations'])
                            for x in result['data']['val']],
            test    = [dict(tokens    = x['tokens'],
                            spans     = x['spans'],
                            relations = x['relations'])
                            for x in result['data']['test']]
        )
   
    elif config.run_type == 'predict':
        dataset = dict(
            predict   = [{'tokens': x['tokens']} for x in result['data']['train']],
        )
    ######################################

    #do schema validity checks
    ######################################
    if 'schema' not in result:
        raise KeyError("The provided JSON file does not contain the required 'schema' key.")
    #Extract the schema
    schema = result['schema']
    #do validity checks
    if 'span_types' not in schema or 'relation_types' not in schema:
        raise ValueError("Invalid schema provided. Ensure it contains 'span_types' and 'relation_types'.")
    #Extract and sort span and relation types, ensuring uniqueness and sorting
    span_types = sorted({x['name'] for x in schema['span_types']})
    rel_types = sorted({x['name'] for x in schema['relation_types']})
    ######################################

    return dict(
        dataset         = dataset,
        span_types      = span_types,
        rel_types       = rel_types
    )
