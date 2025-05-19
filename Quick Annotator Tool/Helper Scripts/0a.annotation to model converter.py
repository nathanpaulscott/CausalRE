import json
import os
import re
import time
import random
import numpy as np
from sklearn.model_selection import train_test_split


'''
This code converts the annotation output to one usable by the model, it does the splits also.
'''

def set_random_seed(seed):
    # Set seed for Python's random module
    random.seed(seed)
    # Set seed for NumPy
    np.random.seed(seed)

seed=129
set_random_seed(seed)

# Adjustable parameters
main_path = 'D:\\A.Nathan\\1a.UWA24-Hons\\Honours Project\\0a.Code\\0a.Nathan Model\\data\\final'
infile = 'conll04_nathan_for_annotation'
obs_to_keep_up_to = 1e6   #1022   #1e6

files_to_merge = [
    #('mixed_pred_final_Excellent', 1e6),
    #('mixed_pred_final_Good(done)', 1e6),
    #('mixed_pred_final_Ave(done)', 1e6),
    #('mixed_pred_final_Poor(done)', 1e6),
    #('mixed_pred_final_Terrible(131)', 131),
]

outfile = infile + f'_for_model_rs{seed}'


#run options:
split_ratios = {'train': 80, 'val': 10, 'test': 10}     #use -1 for test if you just want to copy the val   
id_to_idx = True        #convert id format to idx format
binarize_types = True    #set the span types to all event and rel types to all causal
remove_obs_w_no_spans = True
remove_short_obs_len = 10       #-1 to not filter
remove_long_obs_len = 200    #-1 to ignore
truncate_long_obs = True
def read_json_file(file_path):
    """Reads a JSON file and returns the data as a Python list."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)



def write_json_file(data, file_path):
    """
    Writes the given data to a JSON file.
    """
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)


def read_text_file(file_path):
    """Reads a text file and returns its content as a string, handling encoding issues."""
    with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
        return f.read()



def process_data(data, seed):
    '''
    read the json, make the splits binarize if required
    
    '''
    #init data_out
    data_out = dict(
        schema = data['schema'],
        data = {}
    )    
    

    if id_to_idx:
        #converts from id format to idx format
        for obs in data['data']:
            if len(obs['spans']) == 0:
                continue
            if 'id' not in obs['spans'][0]:
                raise Exception('there are no unique ids, exiting....')
            #make the id to idx mapping
            id_to_idx_map = {span['id']: i for i, span in enumerate(obs['spans'])}
            #apply to teh rel heads/tails
            obs['relations'] = [{**rel, 'head': id_to_idx_map[rel['head']], 'tail': id_to_idx_map[rel['tail']]} for rel in obs['relations']]
            #remove the id from teh spans
            obs['spans'] = [{k: v for k, v in span.items() if k != 'id'} for span in obs['spans']]
            obs['relations'] = [{k: v for k, v in rel.items() if k != 'id'} for rel in obs['relations']]

    if binarize_types:
        data_out['schema'] = dict(
            span_types = [dict(name = "event", color = "rgba(255, 87, 51, 0.3)")],
            relation_types = [dict(name = "causal", color = "rgba(255, 87, 51, 0.3)")]
        )
        #changes the types to just event and causal
        for obs in data['data']:
            obs['spans'] = [{**x, 'type':'event'} for x in obs['spans']]
            obs['relations'] = [{**x, 'type': 'causal'} for x in obs['relations']]

    #remove unwanted obs
    data['data'] = [x for i, x in enumerate(data['data']) if i <= obs_to_keep_up_to]

    #remove obs with no spans
    if remove_obs_w_no_spans:
        data['data'] = [x for x in data['data'] if len(x['spans']) != 0]

    #remove short obs len
    if remove_short_obs_len != -1:
        data['data'] = [x for x in data['data'] if len(x['tokens']) > remove_short_obs_len]

    #remove long obs len or truncate if needed
    if remove_long_obs_len != -1 and not truncate_long_obs:
        data['data'] = [x for x in data['data'] if len(x['tokens']) < remove_long_obs_len]
    elif remove_long_obs_len != -1 and truncate_long_obs:
        #do smart truncation
        for obs in data['data']:
            if len(obs['tokens']) > remove_long_obs_len:
                # Look for last '.' before the limit
                try:
                    trunc_index = max(i for i, tok in enumerate(obs['tokens'][:remove_long_obs_len]) if tok == '.')
                    obs['tokens'] = obs['tokens'][:trunc_index + 1]
                except ValueError:
                    # No '.' found before limit, do hard truncation
                    obs['tokens'] = obs['tokens'][:remove_long_obs_len]

    #split data
    # Convert percentages to proportions
    train_ratio = split_ratios['train'] / 100
    val_ratio = split_ratios['val'] / 100
    if train_ratio == 1:
        #this is a special case, where want just one split
        data_out['data']['train'] = data['data']
        data_out['data']['val'] = []
        data_out['data']['test'] = []

    elif split_ratios['test'] != -1:
        if 1 - train_ratio - val_ratio > 0:
            test_ratio = split_ratios['test'] / 100
        elif split_ratios['test'] != -1:
            test_ratio = 1 - train_ratio - val_ratio
        #First split: train vs (val + test)
        train_data, temp_data = train_test_split(data['data'], test_size=(1 - train_ratio), random_state=seed)
        # Adjust val/test ratios relative to temp_data
        val_adjusted = val_ratio / (val_ratio + test_ratio)
        val_data, test_data = train_test_split(temp_data, test_size=(1 - val_adjusted), random_state=seed)
        data_out['data']['train'] = train_data
        data_out['data']['val'] = val_data
        data_out['data']['test'] = test_data

    else:
        #split: train vs (val + test)
        try:
            train_data, val_data = train_test_split(data['data'], test_size=(1 - train_ratio), random_state=seed)
        except Exception as e:
            pass
        data_out['data']['train'] = train_data
        data_out['data']['val'] = val_data
        data_out['data']['test'] = [x for x in val_data]
       
    return data_out




def main():
    if len(files_to_merge) > 0:
        data = {'schema': {}, 'data': []}
        for file, limit in files_to_merge:
            input_path = f"{main_path}/{file}.json"
            # Read data from JSON file
            incoming = read_json_file(input_path)
            if data['schema'] == {}:
                data['schema'] = incoming['schema']
            data_to_add = [x for i,x in enumerate(incoming['data']) if i <= limit]
            data['data'].extend(data_to_add)
    else:
        # Path to the JSON file to read
        input_path = main_path + '/' + infile + '.json'
        # Read data from JSON file
        data = read_json_file(input_path)
    
    # Path to the JSON file to write
    output_path = main_path + '/' + outfile + '.json'

    # Process the data
    data_out = process_data(data, seed)
    
    # Write processed data to a new JSON file
    write_json_file(data_out, output_path)




if __name__ == "__main__":
    main()