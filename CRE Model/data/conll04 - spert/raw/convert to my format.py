import json



def open_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def convert_to_schema(data, output=None):
    if output is None:
        output = {"schema": {"span_types": [], "relation_types": []}}
    else:
        output['schema'] = {"span_types": [], "relation_types": []}

    for item in data['entities'].keys():
        output['schema']['span_types'].append({"name": item, "color": "rgba(255,255,255,0.3)"})

    for item in data['relations'].keys():
        output['schema']['relation_types'].append({"name": item, "color": "rgba(255,255,255,0.3)"})
    
    return output



def convert_to_split(data, split, output=None):
    if output is None:
        output = {'data': {split: []}}
    else:
        if 'data' in output:
            output['data'][split] = []
        else:
            output['data'] = {}
            output['data'][split] = []

    for item in data:
        obs = {}
        obs['tokens'] = item['tokens']
        obs['spans'] = item['entities']
        obs['relations'] = item['relations']
        output['data'][split].append(obs)

    return output





path = 'D:\\A.Nathan\\1a.UWA24-Hons\\Honours Project\\0a.Code\\0a.Nathan Model\\data\\conll04 - spert'

filename = 'conll04_types'
file_path = f"{path}\\{filename}.json"
data_types = open_json(file_path)

filename = 'conll04_train'
file_path = f"{path}\\{filename}.json"
data_train = open_json(file_path)

filename = 'conll04_dev'
file_path = f"{path}\\{filename}.json"
data_dev = open_json(file_path)

filename = 'conll04_test'
file_path = f"{path}\\{filename}.json"
data_test = open_json(file_path)

output = convert_to_schema(data_types)
output = convert_to_split(data_train, 'train', output)
output = convert_to_split(data_dev, 'val',output)
output = convert_to_split(data_test, 'test',output)

file_path = f"{path}\\conll04_nathan.json"
with open(file_path, 'w') as file:
    json.dump(output, file, indent=4)