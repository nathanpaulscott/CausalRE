import json
from pathlib import Path
from collections import Counter
from typing import List, Dict, Tuple

def load_predictions(file_path):
    """Load predictions from a JSON file."""
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)


def calculate_span_overlap_percentage(span1: Dict, span2: Dict) -> float:
    """
    Calculate the percentage overlap between two spans relative to the first span (labeled span).

    Args:
        span1 (Dict): The labeled span with 'start' and 'end' keys.
        span2 (Dict): The predicted span with 'start' and 'end' keys.

    Returns:
        float: The percentage of the labeled span (span1) that is overlapped by the predicted span (span2).
    """
    start1, end1 = span1['start'], span1['end']
    start2, end2 = span2['start'], span2['end']

    # Calculate the intersection
    overlap_start = max(start1, start2)
    overlap_end = min(end1, end2)
    overlap_length = max(0, overlap_end - overlap_start)
    # Calculate the union
    union_start = min(start1, start2)
    union_end = max(end1, end2)
    union_length = union_end - union_start
    # Calculate the overlap percentage relative to the union of the spans
    return (overlap_length / union_length) * 100 if union_length > 0 else 0


def calculate_rel_overlap_percentage(rel1: Dict, rel2: Dict) -> float:
    """
    Calculate the overlap percentage for relationships based on head-to-head and tail-to-tail overlaps.

    Args:
        rel1 (Dict): The labeled relationship with 'head' and 'tail' keys.
        rel2 (Dict): The predicted relationship with 'head' and 'tail' keys.

    Returns:
        float: The best overlap percentage between the corresponding spans of two relationships.
    """
    head_overlap = calculate_span_overlap_percentage(rel1['head'], rel2['head'])
    tail_overlap = calculate_span_overlap_percentage(rel1['tail'], rel2['tail'])
    # Combine the overlaps by multiplying to reflect combined accuracy
    return head_overlap * tail_overlap / 100



def mark_spans(tokens, spans, prefix):
    for i, span in enumerate(spans):
        idx = span['start']
        tokens[idx] = f'<{prefix}{i}>{tokens[idx]}'
        idx = span['end'] - 1
        tokens[idx] = f'{tokens[idx]}</{prefix}{i}>'


def get_obs_info(obs):
    seq_len = len(obs['tokens'])
    
    num_spans = len(obs['span_labels'])
    num_rels = len(obs['rel_labels'])
    
    mean_span_width = []
    for label in obs['span_labels']:
        mean_span_width.append(label['end'] - label['start'])
    mean_span_width = sum(mean_span_width) / len(mean_span_width) if len(mean_span_width) > 0 else -1
    
    mean_rel_width = []
    mean_rel_width_balance = []
    for label in obs['rel_labels']:
        head_width = label['head']['end'] - label['head']['start']
        tail_width = label['tail']['end'] - label['tail']['start']
        mean_rel_width.append(head_width + tail_width)
        mean_rel_width_balance.append(min(head_width, tail_width)/max(head_width, tail_width))
    mean_rel_width = sum(mean_rel_width) / len(mean_rel_width) if len(mean_rel_width) > 0 else -1
    mean_rel_width_balance = sum(mean_rel_width_balance) / len(mean_rel_width_balance) if len(mean_rel_width_balance) > 0 else -1

    return dict(
        seq_len     = seq_len,
        n_spans     = num_spans,
        n_rels      = num_rels,
        n_spnsp = len(obs.get('span_preds', [])),
        n_rlsp  = len(obs.get('rel_preds', [])),
        span_w      = mean_span_width,
        rel_w       = mean_rel_width,
        rel_bal     = mean_rel_width_balance
    )



def compute_match_rates(data: List[Dict], thresholds) -> dict:
    """Compute match rates considering both directions of matching and classify sequences."""
    output = dict(Excellent=[], Good=[], Ave=[], Poor=[], Terrible=[])

    for obs in data:
        tokens = obs['tokens']
        span_labels = obs['span_labels']
        span_preds = obs['span_preds']
        rel_labels = obs['rel_labels']
        rel_preds = obs['rel_preds']
        #get some info
        info_dict = get_obs_info(obs)

        # Span match calculations
        if span_labels and span_preds:
            label_to_pred_spans = [max((calculate_span_overlap_percentage(label, pred) for pred in span_preds), default=0) for label in span_labels]
            pred_to_label_spans = [max((calculate_span_overlap_percentage(pred, label) for label in span_labels), default=0) for pred in span_preds]
            span_match_rate = (sum(label_to_pred_spans) / len(label_to_pred_spans) + sum(pred_to_label_spans) / len(pred_to_label_spans)) / 2
        elif span_labels and not span_preds:
            span_match_rate = 0  # Labels but no predictions, 0% match rate
        elif span_preds and not span_labels:
            span_match_rate = 0  # Predictions but no labels, 0% match rate
        else:
            span_match_rate = 100  # No labels and no predictions, consider 100% match rate

        # Relationship match calculations
        if rel_labels and rel_preds:
            label_to_pred_rels = [max((calculate_rel_overlap_percentage(label, pred) for pred in rel_preds), default=0) for label in rel_labels]
            pred_to_label_rels = [max((calculate_rel_overlap_percentage(pred, label) for label in rel_labels), default=0) for pred in rel_preds]
            rel_match_rate = (sum(label_to_pred_rels) / len(label_to_pred_rels) + sum(pred_to_label_rels) / len(pred_to_label_rels)) / 2
        elif rel_labels and not rel_preds:
            rel_match_rate = 0
        elif rel_preds and not rel_labels:
            rel_match_rate = 0
        else:
            #I adjust the rel match rate to be ignored if the span math rate is not good, only set to 100% on on labels no preds if the span match rate is ave or above
            rel_match_rate = 100 if span_match_rate > thresholds['Ave'] else -1  # No labels and no predictions, consider 100% match rate

        # Calculate combined match rate based on available data
        rates = [rate for rate in [span_match_rate, rel_match_rate] if rate is not None and rate != -1]
        if rates:
            combined_match_rate = sum(rates) / len(rates)
        else:
            combined_match_rate = -1  # No data to combine

        # Mark spans and format the sequence
        mark_spans(tokens, span_labels, 'L')
        mark_spans(tokens, span_preds, 'P')

        sequence = ' '.join(tokens)
        obs_dict = dict(
            seq = sequence, 
            rate = [span_match_rate, rel_match_rate, combined_match_rate],
        )

        #add the info dict
        obs_dict = obs_dict | info_dict

        # Classify observations based on thresholds
        if combined_match_rate >= thresholds['Excellent']:
            output['Excellent'].append(obs_dict)
        elif thresholds['Good'] <= combined_match_rate < thresholds['Excellent']:
            output['Good'].append(obs_dict)
        elif thresholds['Ave'] <= combined_match_rate < thresholds['Good']:
            output['Ave'].append(obs_dict)
        elif thresholds['Poor'] <= combined_match_rate < thresholds['Ave']:
            output['Poor'].append(obs_dict)
        else:
            output['Terrible'].append(obs_dict)

    #sort by combined rate desc
    output['Excellent'] = sorted(output['Excellent'], key=lambda x: x['rate'][2], reverse=True)
    output['Good'] = sorted(output['Good'], key=lambda x: x['rate'][2], reverse=True)
    output['Ave'] = sorted(output['Ave'], key=lambda x: x['rate'][2], reverse=True)
    output['Poor'] = sorted(output['Poor'], key=lambda x: x['rate'][2], reverse=True)
    output['Terrible'] = sorted(output['Terrible'], key=lambda x: x['rate'][2], reverse=True)
    return output




def output_matches_to_file_and_console(result, path, console=False):
    with open(path, 'w', encoding='utf-8') as file:
        for k, v in result.items():
            # Print and write column titles and counts
            header = f"\n{k} Matches: {len(result[k])}\n"
            # Update header to include extra keys right after rates
            extra_keys = ['seq_len', 'n_spans', 'n_spnsp', 'n_rels', 'n_rlsp', 'span_w', 'rel_w', 'rel_bal']
            header += f"{'Id':<6}{'Span    | Rel     | Comb':<27} | " + ' | '.join(f"{key:>7}" for key in extra_keys) + f" | {'Sequence'}\n"

            if console:
                print(header)
            file.write(header)

            for i, item in enumerate(v):
                # Format rate values
                rate_str = ' | '.join([f"{x:<7.2f}" for x in item['rate']])
                # Format extra data
                extra_data_str = ' | '.join(f"{item.get(key, 0):>7.2f}" for key in extra_keys)  # Right-align and format extra keys
                # Prepare sequence
                sequence = item['seq']
                
                # Format the complete line for output
                line = f"{i:<6}{rate_str} | {extra_data_str} | {sequence}\n"  # Note the rearranged order

                if console:
                    print(line)
                file.write(line)


def process_preds(infile, outfile):
    thresholds = dict(
        Excellent = 95,
        Good = 80,
        Ave = 50,
        Poor = 20,
        Terrible = 0
    )
    data = load_predictions(infile)
    result = compute_match_rates(data, thresholds)
    output_matches_to_file_and_console(result, outfile, console=False)




if __name__ == "__main__":
    ##################################################
    folder = r'D:\A.Nathan\1a.UWA24-Hons\Honours Project\0a.Code\0a.Nathan Model\preds'
    infile_name = 'pred_best.json'
    #infile_name = 'pred_results_model-BECO-tths-nathan-max_temp_tf.json'
    #infile_name = 'pred_results_model-bfhs-nathan-max_temp_tf.json'
    ##################################################
    outfile_name = infile_name[:-4] + '_analysis.txt'
    infile = str(Path(folder + '/' + infile_name))
    outfile = str(Path(folder + '/' + outfile_name))

    process_preds(infile, outfile)




'''
I need to break down some quick classification stats on the dataset here....
So you need to make these features for each obs
NOTE: some of these features can be auto generated, but the key ones need to be human generated, so this analysis can not be done quickly


obs level Sequence features
------------
(auto) seq_len:    how long in tokens
(auto) sent_cnt:   how many sentences in the seq

obs level Span features
----------
(auto) num_spans:  number of spans
(auto) mean_span_width:    mean width of spans
(human) simple_span_cnt: number of spans with simple structure, i.e. the span boundaries are clear and unabiguous, typically events with entity format, but can be trigger-arg if they are clear, usually with minimal args
(human) complex_span_cnt: number of spans with complex structure, i.e. the span boundaries are ambiguous, typically events with trigger-arg format and other more complex patterns

obs level Rel features
----------
(auto) num_rels:   number of rels
(auto) mean_rel_width: mean width of rels (rel width = sum of each component span width)
(human) rels_w_explicit_trigger:   number of rels with an explicit causal trigger
(human) rels_w_implicit_trigger:   number of rels with an implicit causal trigger
(human) rels_w_simple_structure:   number of rels with a simple structure, i.e. spans are close, causality trigger is between them with minimal noise
(human) rels_w_complex_structure:  number of rels with a complex structure, i.e. spans are far, lots of noise language, causality may not be between them

'''





