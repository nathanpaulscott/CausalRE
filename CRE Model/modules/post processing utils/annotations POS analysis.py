import json
import os
import re
import time
import random
import spacy
nlp = spacy.load("en_core_web_sm")
import numpy as np
from collections import defaultdict

import matplotlib.pyplot as plt




# Adjustable parameters
# File paths
#main_path = 'D:\\A.Nathan\\1a.UWA24-Hons\\Honours Project\\0a.Code\\0a.Nathan Model\data\\new causal processed mar25\\maven\\old'
#infile = 'maven - data - short - triggers_for_annotation'

#main_path = 'D:\\A.Nathan\\1a.UWA24-Hons\\Honours Project\\0a.Code\\0a.Nathan Model\\data\\final'
#infile = 'mixed_final_for_annotation'
#infile = 'model_data_semeval_for_annotation'

main_path = 'D:\\A.Nathan\\1a.UWA24-Hons\\Honours Project\\0a.Code\\0a.Nathan Model\\data\\unicausal'
#infile = 'model_data_ctb_for_annotation'
#infile = 'model_data_because_for_annotation'
infile = 'model_data_altlex_for_annotation'

#main_path = 'D:\\A.Nathan\\1a.UWA24-Hons\\Honours Project\\0a.Code\\0a.Nathan Model\\data\\conll04 - spert'
#infile = 'conll04_nathan_for_annotation'

obs_limit = 226   #1e6


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




def analyze_data(input_data):
    span_lengths = []
    pos_counts_in_spans = defaultdict(int)
    pos_counts_in_relations = defaultdict(int)
    total_relations = 0
    obs_with_relations = 0
    rel_context_lengths = []
    win = 10

    filtered_data = [x for i,x in enumerate(input_data['data']) if i <= obs_limit]

    for i, obs in enumerate(filtered_data):
        if i % 100 == 0:
            print(f'doing obs {i} of {len(filtered_data)}')

        # Process spans
        for span in obs['spans']:
            tokens = obs['tokens'][span['start']:span['end']]
            span_length = len(tokens)
            span_lengths.append(span_length)

            doc = nlp(' '.join(tokens))
            for token in doc:
                pos_counts_in_spans[token.pos_] += 1

        # Process relations
        if obs['relations']:
            obs_with_relations += 1
            total_relations += len(obs['relations'])

            '''
            for relation in obs['relations']:
                head_span = next(span for span in obs['spans'] if span['id'] == relation['head'])
                tail_span = next(span for span in obs['spans'] if span['id'] == relation['tail'])
                rel_context = obs['tokens'][tail_span['end']:head_span['start']]
                rel_context_lengths.append(len(rel_context))
                doc = nlp(' '.join(rel_context))
                for token in doc:
                    pos_counts_in_relations[token.pos_] += 1
            '''

            for relation in obs['relations']:
                head_span = next(span for span in obs['spans'] if span['id'] == relation['head'])
                tail_span = next(span for span in obs['spans'] if span['id'] == relation['tail'])

                # Sort spans to get the correct order for context extraction
                first_span, last_span = sorted([head_span, tail_span], key=lambda x: x['start'])

                # Define the extended context range
                context_start = max(0, first_span['start'] - win)
                context_end = min(len(obs['tokens']), last_span['end'] + win)

                # Extract the context before the first span, between spans, and after the last span
                before_first_span = obs['tokens'][context_start:first_span['start']]
                between_spans = [] if first_span['end'] >= last_span['start'] else obs['tokens'][first_span['end']:last_span['start']]
                after_last_span = obs['tokens'][last_span['end']:context_end]

                # Combine all parts of the context
                rel_context = before_first_span + between_spans + after_last_span
                rel_context_lengths.append(len(rel_context))

                doc = nlp(' '.join(rel_context))
                for token in doc:
                    pos_counts_in_relations[token.pos_] += 1



    # Calculate mean span width
    mean_span_width = round(np.mean(span_lengths), 4)
    mean_rel_context_width = round(np.mean(rel_context_lengths), 4)

    total_span_tokens = sum(pos_counts_in_spans.values())
    total_rel_tokens = sum(pos_counts_in_relations.values())

    mean_pos_perc_per_span = dict(sorted(
        ((pos, round((counts / total_span_tokens) * 100, 4)) for pos, counts in pos_counts_in_spans.items()),
        key=lambda item: item[1], reverse=True))

    mean_pos_perc_per_relation = dict(sorted(
        ((pos, round((counts / total_rel_tokens) * 100, 4)) for pos, counts in pos_counts_in_relations.items()),
        key=lambda item: item[1], reverse=True))


    '''
    # Calculate and sort mean POS counts per span and per relation context
    mean_pos_perc_per_span = dict(sorted(
        ((pos, round(counts / len(input_data['data']), 6)) for pos, counts in pos_counts_in_spans.items()),
        key=lambda item: item[1], reverse=True))

    mean_pos_perc_per_relation = dict(sorted(
        ((pos, round(counts / len(input_data['data']), 6)) for pos, counts in pos_counts_in_relations.items()),
        key=lambda item: item[1], reverse=True))
    '''
    
    # Ratio of observations with at least one relation
    ratio_with_relations = round(obs_with_relations / len(input_data['data']), 6)

    # Mean number of relations per observation
    mean_relations_per_obs = round(total_relations / len(input_data['data']), 6)

    return dict(
        mean_span_width            = mean_span_width,
        mean_rel_context_width     = mean_rel_context_width, 
        mean_pos_perc_per_span     = mean_pos_perc_per_span, 
        mean_pos_perc_per_relation = mean_pos_perc_per_relation, 
        ratio_with_relations       = ratio_with_relations, 
        mean_relations_per_obs     = mean_relations_per_obs
    )





def plot(data):
    key_pos_types = ['NOUN', 'VERB', 'ADJ', 'PROPN', 'ADP', 'DET', 'PRON']
    # Filter data to include only key POS types
    filtered_span_data = {pos: data['mean_pos_perc_per_span'].get(pos, 0) for pos in key_pos_types}
    filtered_rel_data = {pos: data['mean_pos_perc_per_relation'].get(pos, 0) for pos in key_pos_types}

    # Sort each filtered data dictionary by value in descending order
    filtered_span_data = dict(sorted(filtered_span_data.items(), key=lambda item: item[1], reverse=True))
    filtered_rel_data = dict(sorted(filtered_rel_data.items(), key=lambda item: item[1], reverse=True))

    # Extract labels and values for plotting
    span_labels, span_values = zip(*filtered_span_data.items())
    rel_labels, rel_values = zip(*filtered_rel_data.items())

    # Calculate positions for each group
    span_positions = np.arange(len(span_labels))  # Positions for the spans group
    group_gap = 1  # Gap between groups
    rel_positions = np.arange(len(rel_labels)) + len(span_labels) + group_gap  # Positions for the relations group

    width = 0.9  # Width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(span_positions, span_values, width, label='Spans')
    rects2 = ax.bar(rel_positions, rel_values, width, label='Relations')

    # Add some text for labels, title, and custom x-axis tick labels, etc.
    ax.set_ylabel('Percentage')
    ax.set_title('Percentage of POS tags in Spans and Relations')
    ax.set_xticks(np.concatenate([span_positions, rel_positions]))
    ax.set_xticklabels(list(span_labels) + list(rel_labels), rotation=45, ha='right')
    ax.legend()

    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}%',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)

    fig.tight_layout()

    plt.show()




def main():
    # Path to the JSON file to read
    input_path = main_path + '/' + infile + '.json'
    # Path to the JSON file to write
    #output_path = main_path + '/' + outfile + '.json'

    # Read data from JSON file
    #'''
    data = read_json_file(input_path)
    
    # Run the analysis
    #analyze_data(data)
    result = analyze_data(data)
    print(f"Mean Span Width: {result['mean_span_width']}")
    print(f"Mean Rel Context Width: {result['mean_rel_context_width']}")
    print("Mean POS Counts per Span:", result['mean_pos_perc_per_span'])
    print("Mean POS Counts per Relation Intervening Tokens:", result['mean_pos_perc_per_relation'])
    print("Ratio of Observations with Relations:", result['ratio_with_relations'])
    print("Mean Number of Relations per Observation:", result['mean_relations_per_obs'])
    '''
    result = dict(
        mean_pos_perc_per_span = {'NOUN': 1.694411, 'VERB': 0.227209, 'PROPN': 0.112625, 'ADJ': 0.056359, 'PUNCT': 0.005972, 'ADV': 0.004852, 'AUX': 0.00168, 'ADP': 0.0014, 'INTJ': 0.0014, 'X': 0.001306, 'NUM': 0.000467, 'CCONJ': 0.000373, 'PRON': 0.000187, 'SCONJ': 9.3e-05, 'DET': 9.3e-05},
        mean_pos_perc_per_relation = {'NOUN': 0.463096, 'ADP': 0.337968, 'DET': 0.25091, 'VERB': 0.249324, 'PUNCT': 0.244005, 'ADJ': 0.166185, 'PRON': 0.139218, 'AUX': 0.100401, 'PROPN': 0.091164, 'CCONJ': 0.085845, 'ADV': 0.058225, 'NUM': 0.031912, 'PART': 0.029113, 'SCONJ': 0.025194, 'SYM': 0.002146, 'X': 0.00056, 'INTJ': 0.000187}
    )
    '''
    plot(result)


    # Write processed data to a new JSON file
    #write_json_file(data_out, output_path)



if __name__ == "__main__":
    main()