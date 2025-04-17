import pandas as pd
import json
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt_tab')




def load_json(input_file):
    with open(input_file, 'r', encoding='utf-8') as file:
        data = json.load(file)

    return data



def save_json(data, output_file):
    with open(output_file, 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=4)



def remove_dataset_duplicates(dataset):
    unique_entries = {}
    for entry in dataset['data']:
        # Convert tokens list to a tuple for immutability and use as a dictionary key
        token_tuple = tuple(entry['tokens'])

        # Check if this token tuple has been seen before
        if token_tuple in unique_entries:
            # Only replace if the current entry has more spans than the one stored
            if len(entry['spans']) > len(unique_entries[token_tuple]['spans']):
                unique_entries[token_tuple] = entry
        else:
            # Store this entry as it's the first occurrence of these tokens
            unique_entries[token_tuple] = entry
    
    #overwrite the dataset
    dataset['data'] = list(unique_entries.values())




def load_data(file_path, delimiter=','):
    try:
        # Load the data
        data = pd.read_csv(file_path, delimiter=delimiter)
        
        # Strip spaces from column names
        data.columns = data.columns.str.strip()

        return data
    except Exception as e:
        print(f"An error occurred: {e}")



def jaccard_similarity(list1, list2):
    set1, set2 = set(list1), set(list2)
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union != 0 else 0



def get_dataset_id(analysis_data, dataset):
    #remove the annoation tags
    analysis_data['Sequence'] = analysis_data['Sequence'].str.replace(r'</?[PL]\d+>', '', regex=True)
    #make class a cat col
    class_order = ['Excellent', 'Good', 'Ave', 'Poor', 'Terrible']
    analysis_data['Class'] = pd.Categorical(analysis_data['Class'], categories=class_order, ordered=True)
    #remove the duplicated sequences and keep the one wiht more spans annaoted
    # Sort the DataFrame by 'Sequence' and then by 'n_spans' in descending order
    analysis_data = analysis_data.sort_values(by=['Sequence', 'n_spans'], ascending=[True, False])
    # Drop duplicates based on 'Sequence' keeping the first entry with the highest 'n_spans'
    analysis_data = analysis_data.drop_duplicates(subset='Sequence', keep='first')
    # Additional sorting by 'Class' and then by 'Comb' descending
    analysis_data = analysis_data.sort_values(by=['Class', 'Comb'], ascending=[True, False])
    # Tokenize sequences in the DataFrame
    analysis_data['Tokenized'] = analysis_data['Sequence'].apply(lambda x: word_tokenize(x))

    # Prepare dataset sequences
    dataset_token_lists = [obs['tokens'] for obs in dataset['data']]

    # Initialize a column for storing dataset IDs
    analysis_data['DatasetID'] = None
    analysis_data['BestMatchScore'] = 0

    # Loop over each row in the DataFrame
    for idx, row in analysis_data.iterrows():
        best_score = 0
        best_match_id = None
        for i, tokens in enumerate(dataset_token_lists):
            score = jaccard_similarity(row['Tokenized'], tokens)
            if score > best_score:
                best_score = score
                best_match_id = i

        # Store the best match ID and its score if it's above a certain threshold
        if best_score > 0.5:  # Adjust threshold as necessary
            analysis_data.at[idx, 'DatasetID'] = best_match_id
            analysis_data.at[idx, 'BestMatchScore'] = best_score
    
    return analysis_data



def update_dataset_classes_and_save(analysis_data, dataset, dataset_path):
    """
    Update the dataset['data'] with the 'Class' from the analysis_data DataFrame based on the DatasetID,
    and save each class into separate JSON files along with the original schema.
    
    :param analysis_data: DataFrame with columns 'DatasetID' and 'Class'
    :param dataset: dictionary with keys 'schema' and 'data' that contains a list of observations
    """
    # Update classes in dataset['data']
    for idx, row in analysis_data.iterrows():
        dataset_id = row['DatasetID']
        class_label = row['Class']
        
        # Ensure the dataset_id is within the range of dataset['data']
        if dataset_id < len(dataset['data']):
            dataset['data'][dataset_id]['Class'] = class_label
        else:
            print(f"DatasetID {dataset_id} is out of range for the dataset.")

    # Split dataset into separate classes and save
    class_datasets = {}
    for item in dataset['data']:
        cls = item.get('Class', 'Unknown')  # Default class if not set
        if cls not in class_datasets:
            class_datasets[cls] = {'schema': dataset['schema'], 'data': []}
        class_datasets[cls]['data'].append(item)
    
    # Save each class dataset to a separate JSON file
    for cls, data in class_datasets.items():
        output_file = f"{dataset_path}_{cls}.json"
        save_json(data, output_file)
        print(f"Data for class {cls} saved to {output_file}")



def find_non_integer_rows(dataframe, column_name):
    # Temporarily convert the column to numeric, non-convertible values become NaN
    temp_column = pd.to_numeric(dataframe[column_name], errors='coerce')

    # Find rows where the column is NaN (these were non-integer values)
    non_integer_rows = dataframe[temp_column.isna()]

    return non_integer_rows



def find_rows_with_keywords(dataframe, column_name, keywords):
    # Create a mask for each keyword
    mask = dataframe[column_name].str.contains(keywords[0], case=False, na=False)
    for keyword in keywords[1:]:
        mask &= dataframe[column_name].str.contains(keyword, case=False, na=False)
    
    # Apply mask to dataframe to filter rows
    filtered_rows = dataframe[mask]
    return filtered_rows


#load the .txt analysis file
analysis_path = 'D:\\A.Nathan\\1a.UWA24-Hons\\Honours Project\\0a.Code\\0a.Nathan Model\\preds'
analysis_filename = 'pred_analysis_baseline_mod'
analysis_file_path = analysis_path + '\\' + f"{analysis_filename}.txt"
analysis_data = load_data(analysis_file_path, '\t')

#load the json dataset
dataset_path = 'D:\\A.Nathan\\1a.UWA24-Hons\\Honours Project\\0a.Code\\0a.Nathan Model\\data\\unicausal'
dataset_filename = 'mixed_pred_1022(back to 935)'
dataset_file_path = dataset_path + '\\' + f"{dataset_filename}.json"
dataset = load_json(dataset_file_path)
#remove_dataset_duplicates(dataset)
#save_json(dataset, dataset_path + '\\' + f"{filename}_deuplicated.json")

#find the obs od in the dataset
analysis_data = get_dataset_id(analysis_data, dataset)

print(analysis_data.head())  # Display the first few rows of the dataframe

#write to disk
analysis_data.to_csv(analysis_path + '\\' + f"{analysis_filename}_processed.txt", index=False)  # Set index=False to exclude row indices from the CSV file
#split the datasetby class and save
update_dataset_classes_and_save(analysis_data, dataset, dataset_path + '\\' + f"{dataset_filename}")

'''
# Assuming 'DatasetID' is the column you added and want to check for non-integers:
#rows with no good match
non_integer_data = find_non_integer_rows(analysis_data, 'DatasetID')
print("Non-Integer Rows:")
print(non_integer_data)

# Find duplicates based on 'DatasetID'
duplicates = analysis_data[analysis_data.duplicated('DatasetID', keep=False)].sort_values(by=['Sequence', 'DatasetID'])
# Show the rows with repeated DatasetID
print(f"Rows with repeated DatasetID: there are {len(duplicates)} rows")
print(duplicates)




# Assuming 'Sequence' is the column you're interested in:
keywords = ["Google", "Nest"]
rows_with_keywords = find_rows_with_keywords(analysis_data, 'Sequence', keywords)
print("Rows containing 'Google' and 'Nest':")
print(rows_with_keywords)
'''
