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
    entities, relations = x['entities'], x['relations']
    B = len(entities)
    relation_with_span = [[] for i in range(B)]
    for i in range(B):
        rel_i = relations[i]
        ent_i = entities[i]
        for rel in rel_i:
            act = (ent_i[rel[0]], ent_i[rel[1]], rel[2])
            relation_with_span[i].append(act)
    return relation_with_span


def get_ground_truth_relations(x, candidate_spans_idx, candidate_span_label):
    """
    As usual the naming is fucking misleading, this guy needs a good talking to!!
    This function actually generates the relation labels tensor for all possible span-pairs derived from candidate_spans_idx
    The output shape is (batch, candidate_span_label.shape[1]**2)   This is the quadratic relation expansion part
    NOTE: this function is really horrible, working in python objects, lots of list searches and bad var naming, the whole thing needs to be redone from the ground up
    
    inputs:
    x => so all inputs
    candidate_spans_idx => the idx tensor of the candidate spans (start, end) tuples for each obs
    candidate_span_label => the labels tensor of the candidate spans
    NOTE: we already have the main span labels, so we do not really need this, but it saves us having to extract it
    
    Outputs:
    relation_classes, which is a tensor of shape (batch, max_top_k**2) with the ground truth label for each candidate relation and -1 if it has no label
    """
    #get dims of the caididate span tensor
    B, max_top_k = candidate_span_label.shape

    #Nathan: make the blank relation_slasses tensor of shape (batch, num_cand_spans, num_cand_spans), to be flattened in the last 2 dims later
    #So, this should be called candidate_rel_label, not relation_classes!!!!
    relation_classes = torch.zeros((B, max_top_k, max_top_k), dtype=torch.long, device=candidate_spans_idx.device)

    # Populate relation classes
    #Nathan: ya, he just fills out the relation_classes tensor based on the relation labels in x['relations'] and x['entities]
    #before getting into this, to me, it seems like he is making it way harder than it needs to be, you already have all the data, you just need some indexing and filtering
    # I think x['entities'] and x['relations'] are still in python object format, he should have already converted this to tensor by now, he is doing it inside the model, no good!!!
    for i in range(B):
        #get the rel and ent labels for the obs
        rel_i = x["relations"][i]
        ent_i = x["entities"][i]

        #these are the head,tail,rel_type lists for the candidate rels
        new_heads, new_tails, new_rel_type = [], [], []

        #Nathan: Loop over the ground truth relation labels lists to extract the rel types, the span start/end bounds from the head/tail idx
        #so the output here is still python objects
        for k in rel_i:
            """
            #he should have better code like this:
            head_span_idx, tail_span_idx, rel_type = k
            head_span_start, head_span_end = ent_i[head_span_idx]
            tail_span_start, tail_span_end = ent_i[tail_span_idx]
            new_heads.append((head_span_start, head_span_end))
            new_tails.append((tail_span_start, tail_span_end))
            new_rel_type.append(rel_type)
            """
            #nathan: read the head span boundary
            heads_i = [ent_i[k[0]][0], ent_i[k[0]][1]]
            #nathan: read the tail span boundary
            tails_i = [ent_i[k[1]][0], ent_i[k[1]][1]]
            #nathan: read the relation type
            type_i = k[2]
            new_heads.append(heads_i)
            new_tails.append(tails_i)
            new_rel_type.append(type_i)

        # Update the original lists
        #Nathan: he is doing absolutely nothing here, just changing the names of the 3 lists!!?!?!?
        heads_, tails_, rel_type = new_heads, new_tails, new_rel_type

        # idx of candidate spans
        #Nathan: converting the cand_spans_idx tensor (tensor of candidate span start/end tuples) for the obs to a list (python object)  why is he staying in python objects at this point???
        cand_i = candidate_spans_idx[i].tolist()

        #Nathan: now he goes through the extracted relation ground truth label data and updates the relation_classes tensor
        #again this loop is full of terrible var naming => eg. flag!!   fucking for what is the flag being used?
        for heads_i, tails_i, type_i in zip(heads_, tails_, rel_type):
            #first he determines if the rel_type is in the rel_to_id mapping, there is no reason it would not be as the mapping was created from the ground truth labels!?!?!?
            #this should not even be needed, just a side effect of bad coding
            #should be known_rel_type_flag or something....
            flag = False
            if isinstance(x["rel_to_id"], dict):
                if type_i in x["rel_to_id"]:
                    flag = True
            elif isinstance(x["rel_to_id"], list):
                if type_i in x["rel_to_id"][i]:
                    flag = True

            #Nathan: this is the key check, he checks if the head_span and tail_span are in the cand_span_idx list
            #terrible way to do it, he is doing a two list searches here, very slow, should have found a smarter way!!!!  (1) heads_i in cand_i and (2) list.index()
            #logic is if the head and tail spans are in teh cand list and the rel type is known (which it should always be), then process it
            if heads_i in cand_i and tails_i in cand_i and flag:   
                #get teh list index of the head and tail span of the relation from the cand list
                idx_head = cand_i.index(heads_i)
                idx_tail = cand_i.index(tails_i)

                #Nathan: here they populate the relation_classes tensor with the candidate relation type read from teh ground thruth labels
                #what a round about horrible way to get there!!!
                if isinstance(x["rel_to_id"], list):
                    relation_classes[i, idx_head, idx_tail] = x["rel_to_id"][i][type_i]
                elif isinstance(x["rel_to_id"], dict):
                    relation_classes[i, idx_head, idx_tail] = x["rel_to_id"][type_i]

    # flat relation classes
    #Nathan: flattens the last 2 dims of the relation_classes so teh tensor is now (batch, max_top_k**2)
    relation_classes = relation_classes.view(-1, max_top_k * max_top_k)

    # put to -1 class where corresponding candidate_span_label is -1 (for both head and tail)
    #bad english, what he means is put the rel type to -1 if EITHER head OR tail span type is -1 (invalid or set to -1 for filtering purposes)
    #first he exacts the head_span_label and tail_Span_label for every element in relation_classes
    #Nathan: due to the way he crated the relation_classes, they are aligned with the candidate_span_label tensors, 
    head_candidate_span_label = candidate_span_label.view(B, max_top_k, 1).repeat(1, 1, max_top_k).view(B, -1)
    tail_candidate_span_label = candidate_span_label.view(B, 1, max_top_k).repeat(1, max_top_k, 1).view(B, -1)
    #then here he sets the relation_classes tensor element to -1 if either of the head or tail span types are -1, i.e. only allowing relations where both head and tail spans have a positive ground truth label in the cand set
    relation_classes.masked_fill_(head_candidate_span_label.view(B, max_top_k * max_top_k) == -1, -1)  # head
    relation_classes.masked_fill_(tail_candidate_span_label.view(B, max_top_k * max_top_k) == -1, -1)  # tail

    return relation_classes




def get_candidates(sorted_idx, tensor_elem, topk=10):
    # sorted_idx [B, num_spans]
    # tensor_elem [B, num_spans, D] or [B, num_spans]

    sorted_topk_idx = sorted_idx[:, :topk]

    if len(tensor_elem.shape) == 3:
        B, num_spans, D = tensor_elem.shape
        topk_tensor_elem = tensor_elem.gather(1, sorted_topk_idx.unsqueeze(-1).expand(-1, -1, D))
    else:
        # [B, topk]
        topk_tensor_elem = tensor_elem.gather(1, sorted_topk_idx)

    return topk_tensor_elem, sorted_topk_idx



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



def import_data(data_path):
    """
    Load a source dataset from a JSON file and extract its schema.

    Args:
        data_path (str): The path to the source JSON file.

    Input JSON Format:
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

    Returns:
        tuple: A tuple containing:
            - data (dict): The dataset without the schema key.
            - schema (dict): The extracted schema.

    Raises:
        KeyError: If the 'schema' key is missing from the JSON.
    """
    #Load the JSON file into a Python object
    result = load_from_json(data_path)
    
    #do validity checks for splits
    if result is None or 'data' not in result or 'train' not in result['data'] or 'val' not in result['data'] or 'test' not in result['data']:
        raise ValueError("Invalid data provided. Ensure it contains 'train', 'val' and 'test' keys")

    #do schema validity checks
    if 'schema' not in result:
        raise KeyError("The provided JSON file does not contain the required 'schema' key.")
    #Extract the schema and remove it from the data
    schema = result['schema']
    #do validity checks
    if 'span_types' not in schema or 'relation_types' not in schema:
        raise ValueError("Invalid schema provided. Ensure it contains 'span_types' and 'relation_types'.")
    #Extract and sort span and relation types, ensuring uniqueness and sorting
    span_types = sorted({x['name'] for x in schema['span_types']})
    rel_types = sorted({x['name'] for x in schema['relation_types']})

    return result['data'], span_types, rel_types