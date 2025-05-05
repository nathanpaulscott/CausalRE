import json
import os
import pandas as pd

def read_csv_file(file_path):
    return pd.read_csv(file_path)

def read_json_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_to_json(data, output_path):
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def find_matching_sequence(tokens, df_texts):
    token_seq = " ".join(tokens).strip()
    for idx, row in df_texts.iterrows():
        if token_seq == row['Source'].strip():
            return idx
    return None

def insert_tags(tokens, spans, prefix):
    for idx, span in enumerate(spans):
        start, end = span['start'], span['end']
        tokens[start] = f"<{prefix}{idx}>" + tokens[start]
        tokens[end-1] = tokens[end-1] + f"</{prefix}{idx}>"
    return tokens

def insert_relation_tags(tokens, rels, prefix):
    for idx, rel in enumerate(rels):
        for role, side in [('H', rel['head']), ('T', rel['tail'])]:
            start, end = side['start'], side['end']
            tokens[start] = f"<{prefix}{idx}{role}>" + tokens[start]
            tokens[end-1] = tokens[end-1] + f"</{prefix}{idx}{role}>"
    return tokens

def apply_inserts(tokens):
    output = []
    for idx, token in enumerate(tokens):
        if token in {'.', ',', '?', '!', ';', ':'} and idx > 0:
            output[-1] = output[-1] + token
        else:
            output.append(token)
    return " ".join(output)

def process_data(json_data, df_texts):
    results = []

    for obs in json_data:
        tokens = obs['tokens']
        match_idx = find_matching_sequence(tokens, df_texts)
        if match_idx is None:
            print("No match found!")
            continue

        span_labels = obs.get('span_labels', [])
        span_preds = obs.get('span_preds', [])
        rel_labels = obs.get('rel_labels', [])
        rel_preds = obs.get('rel_preds', [])

        tokens_span_labels = tokens.copy()
        tokens_span_preds = tokens.copy()
        tokens_rel_labels = tokens.copy()
        tokens_rel_preds = tokens.copy()

        tokens_span_labels = insert_tags(tokens_span_labels, span_labels, 'L')
        tokens_span_preds = insert_tags(tokens_span_preds, span_preds, 'P')
        tokens_rel_labels = insert_relation_tags(tokens_rel_labels, rel_labels, 'R')
        tokens_rel_preds = insert_relation_tags(tokens_rel_preds, rel_preds, 'R')

        obs['annotated_sequence'] = {
            'span_labels': apply_inserts(tokens_span_labels),
            'span_preds': apply_inserts(tokens_span_preds),
            'rel_labels': apply_inserts(tokens_rel_labels),
            'rel_preds': apply_inserts(tokens_rel_preds)
        }

        # Add Preferred, Annotation Issues, and Comment fields
        obs['Preferred'] = df_texts.loc[match_idx, 'Preferred'] if 'Preferred' in df_texts.columns else None
        obs['Annotation Issues'] = df_texts.loc[match_idx, 'Annotation Issues'] if 'Annotation Issues' in df_texts.columns else None
        obs['Comment'] = df_texts.loc[match_idx, 'Comment'] if 'Comment' in df_texts.columns else None

        results.append(obs)

    return results

def main():
    main_path = "D:\\A.Nathan\\1a.UWA24-Hons\\Honours Project\\0a.Code\\0a.Nathan Model\\0a.final analysis\\preds"
    json_file = "pred_best_final.json"
    csv_file = "best_final review.csv"

    json_path = os.path.join(main_path, json_file)
    csv_path = os.path.join(main_path, csv_file)

    json_data = read_json_file(json_path)
    df = read_csv_file(csv_path)

    result = process_data(json_data, df)

    output_path = os.path.join(main_path, json_file.replace(".json", "_processed.json"))
    save_to_json(result, output_path)

if __name__ == "__main__":
    main()