'''
Quick and dirty code to merge 3 separate train, val, test json files to one for my model
'''



import json

def open_json(input_file):
    # Read JSON file
    try:
        with open(input_file, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        raise Exception(f"Error: Could not find {input_file}")
    except json.JSONDecodeError:
        raise Exception(f"Error: Invalid json in {input_file}")
        return


def filter_keys(dict, keys):
    return {k:v for k,v in dict.items() if k in keys}


# Example filter functions:
def filter(train, val, test):
    output = {}
    print(train.keys())
    output['schema'] = train['schema']
    output['data'] = {}
    output['data']['train'] = train['data']
    output['data']['val'] = val['data']
    output['data']['test'] = test['data']

    keys2keep = ['tokens','spans','relations']
    output['data']['train'] = [filter_keys(x, keys2keep) for x in output['data']['train']]
    output['data']['val'] = [filter_keys(x, keys2keep) for x in output['data']['val']]
    output['data']['test'] = [filter_keys(x, keys2keep) for x in output['data']['test']]

    return output


# Example usage:
if __name__ == "__main__":
    # Example data processing
    main_path = 'D:/A.Nathan/1a.UWA24-Hons/Honours Project/0a.Code/0a.Nathan Model/data'
    input_file_train = f'{main_path}/import_file_annotated_altlex_train_combined.json'
    input_file_test = f'{main_path}/import_file_annotated_altlex_test_combined.json'
    output_file = f'{main_path}/output.json'


    data_train = open_json(input_file_train)
    #data_val = open_json(input_file_val)
    data_test = open_json(input_file_test)

    #Apply filter
    data = filter(data_train, data_test, data_test)

    #Save filtered data
    try:
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=4)
        print(f"Successfully saved filtered data to {output_file}")
    except:
        print(f"Error: Could not write to {output_file}")
