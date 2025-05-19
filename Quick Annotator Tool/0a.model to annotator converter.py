import json
import os
import re
import time
import random

'''
This code converts the molel data format to the annotator format, it removes the splits also.
'''



# Adjustable parameters
main_path = 'D:\\A.Nathan\\1a.UWA24-Hons\\Honours Project\\0a.Code\\0a.Nathan Model\\data\\unicausal'
infile = "model_data_because"



#main_path = 'D:\\A.Nathan\\1a.UWA24-Hons\\Honours Project\\0a.Code\\0a.Nathan Model\\data\\conll04 - spert'
#infile = 'conll04_nathan'
outfile = infile + '_for_annotation'


#run options:
splits = ['train', 'val']  #, 'test']   #which splits to merge, eg. if test is just a copy of val, then do not include it here!!!
idx_to_id = True        #convert idx format to id format
binarize_types = True    #set the span types to all event and rel types to all causal



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




def process_data(data):
    '''
    read json
    schema is unchanged
    get the required splits ['train'. 'val', 'test'] modify if required

    merge the splits, add a key n each obs indicating the split => call it 'split': 'train' etc...

    then add unqiue ids to spans
    i.e. id: span list idx
    add unique ids to rels
    i.e id: rel list idx
    NOTE: if you just use the span list idx, you will not have to modify the head and tail values
    '''
    #init data_out
    data_out = dict(
        schema = data['schema'],
        data = []
    )    
    
    #merge splits
    for split in splits:
        data_out['data'].extend([{**x, 'split_orig': split} for x in data['data'][split]])

    if idx_to_id:
        #converts from idx format (using list indices to id format (using unique span and rel ids)
        for obs in data_out['data']:
            if len(obs['spans']) > 0 and 'id' in obs['spans'][0]:
                raise Exception('there are already unique ids, you can not do this operation, think carefully about it!!!')
            obs['spans'] = [{**x, 'id': 'E' + str(i)} for i, x in enumerate(obs['spans'])]
            obs['relations'] = [{**x, 'head': 'E' + str(x['head']), 'tail': 'E' + str(x['tail']), 'id': 'R' + str(i)} for i, x in enumerate(obs['relations'])]

    if binarize_types:
        data_out['schema'] = dict(
            span_types = [dict(name = "event", color = "rgba(255, 87, 51, 0.3)")],
            relation_types = [dict(name = "causal", color = "rgba(255, 87, 51, 0.3)")]
        )
        #changes the types to just event and causal
        for obs in data_out['data']:
            obs['spans'] = [{**x, 'type':'event'} for x in obs['spans']]
            obs['relations'] = [{**x, 'type': 'causal'} for x in obs['relations']]


    return data_out




def main():
    # Path to the JSON file to read
    input_path = main_path + '/' + infile + '.json'
    # Path to the JSON file to write
    output_path = main_path + '/' + outfile + '.json'

    # Read data from JSON file
    data = read_json_file(input_path)
    
    # Process the data
    data_out = process_data(data)
    
    # Write processed data to a new JSON file
    write_json_file(data_out, output_path)




if __name__ == "__main__":
    main()