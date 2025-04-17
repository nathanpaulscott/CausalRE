import json

def convert_predictions_to_annotations(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as file:
        data = json.load(file)

    annotation_data = {
        'schema': {},  # This will be filled manually as mentioned
        'data': []
    }

    for obs in data:
        annotation_obs = {
            'tokens': obs['tokens'],
            'spans': [],
            'relations': []
        }

        # Mapping spans with simple IDs
        span_id_map = {}
        for i, span in enumerate(obs.get('span_preds', [])):
            span_id = f"S{i}"
            span_id_map[(span['start'], span['end'], span['type'])] = span_id
            span_copy = span.copy()
            span_copy['id'] = span_id
            annotation_obs['spans'].append(span_copy)

        # Processing relations and mapping with span IDs
        for j, rel in enumerate(obs.get('rel_preds', [])):
            rel_id = f"R{j}"
            head_span_key = (rel['head']['start'], rel['head']['end'], rel['head']['type'])
            tail_span_key = (rel['tail']['start'], rel['tail']['end'], rel['tail']['type'])

            # Ensure head and tail spans exist, or add them
            for key in [head_span_key, tail_span_key]:
                if key not in span_id_map:
                    new_span_id = f"S{len(span_id_map)}_unk"
                    span_id_map[key] = new_span_id
                    annotation_obs['spans'].append({'start': key[0], 'end': key[1], 'type': key[2], 'id': new_span_id})

            annotation_obs['relations'].append({
                'id': rel_id,
                'type': rel['type'],
                'head': span_id_map[head_span_key],
                'tail': span_id_map[tail_span_key]
            })

        annotation_data['data'].append(annotation_obs)

    with open(output_file, 'w', encoding='utf-8') as file:
        json.dump(annotation_data, file, indent=4)

# Usage
base_path = 'D:\\A.Nathan\\1a.UWA24-Hons\\Honours Project\\0a.Code\\0a.Nathan Model\\preds'
input_name = 'pred_baseline'
input_file_path = base_path + '\\' + f'{input_name}.json'
output_file_path = base_path + '\\' + f'{input_name}_for_annotation.json'
convert_predictions_to_annotations(input_file_path, output_file_path)
