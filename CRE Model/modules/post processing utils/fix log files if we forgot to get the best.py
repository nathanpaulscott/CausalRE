import os
import re
from pathlib import Path

def make_model_save_score(result):
    # Span metrics
    p = result['span_metrics']['precision']
    r = result['span_metrics']['recall']
    f1 = result['span_metrics']['f1']
    balance = (min(p, r) / max(p, r)) ** 0.25 if max(p, r) > 0 else 0
    span_score = f1 * balance

    # Rel metrics
    p = result['rel_metrics']['precision']
    r = result['rel_metrics']['recall']
    f1 = result['rel_metrics']['f1']
    balance = (min(p, r) / max(p, r)) ** 0.25 if max(p, r) > 0 else 0
    rel_score = f1 * balance

    return (span_score + rel_score) / 2

def extract_metrics_from_line(line):
    def extract(m):
        return float(m.group(1)) if m else 0.0

    span_match = re.search(r'Span:.*?P: ([\d.]+)%.*?R: ([\d.]+)%.*?F1: ([\d.]+)%', line)
    rel_match = re.search(r'Rel:.*?P: ([\d.]+)%.*?R: ([\d.]+)%.*?F1: ([\d.]+)%', line)
    span = {
        'precision': extract(re.search(r'P: ([\d.]+)%', span_match.group(0))) if span_match else 0.0,
        'recall': extract(re.search(r'R: ([\d.]+)%', span_match.group(0))) if span_match else 0.0,
        'f1': extract(re.search(r'F1: ([\d.]+)%', span_match.group(0))) if span_match else 0.0,
    }
    rel = {
        'precision': extract(re.search(r'P: ([\d.]+)%', rel_match.group(0))) if rel_match else 0.0,
        'recall': extract(re.search(r'R: ([\d.]+)%', rel_match.group(0))) if rel_match else 0.0,
        'f1': extract(re.search(r'F1: ([\d.]+)%', rel_match.group(0))) if rel_match else 0.0,
    }
    return {'span_metrics': span, 'rel_metrics': rel}



def extract_eval_block_from_line(line):
    block = []
    block.append("0000-00-00 00:00:00,000 - INFO - root - Eval_type: final_val  ")
    block.append("-------------------------------")

    for key in ['Span', 'Rel', 'Rel_mod', 'Span_l', 'Rel_l', 'Rel_mod_l']:
        m = re.search(fr'{key}:\s*\(S: (\d+),\s*P: ([\d.]+)%,\s*R: ([\d.]+)%,\s*F1: ([\d.]+)%\)', line)
        if m:
            s, p, r, f1 = m.groups()
            block.append(f"{key+':':<11} S: {s}\tP: {p}%\tR: {r}%\tF1: {f1}%")
    block.append("-------------------------------")
    return '\n'.join(block)



def process_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    eval_lines = [line for line in lines if 'train loss mean' in line and 'Span:' in line and 'Rel:' in line]
    
    best_score = -1
    best_line = None

    for line in eval_lines:
        metrics = extract_metrics_from_line(line)
        score = make_model_save_score(metrics)
        if score > best_score:
            best_score = score
            best_line = line

    if best_line:
        eval_block = extract_eval_block_from_line(best_line)
        output_path = file_path.with_name(file_path.name.replace('_train_fix.log', '_train_fixed.log'))

        with open(output_path, 'w', encoding='utf-8') as f_out:
            f_out.writelines(lines)
            f_out.write('\n')
            f_out.write("2025-04-15 00:00:00,000 - INFO - root - Eval_type: final_val \n")
            f_out.write("-------------------------------\n")
            for line in eval_block.split('\n')[2:-1]:  # skip Eval_type header and footer dashes
                f_out.write(line + '\n')
            f_out.write(eval_block + '\n')

        print(f"Processed: {output_path}")

def main():
    path = Path("D:\\A.Nathan\\1a.UWA24-Hons\\Honours Project\\0a.Code\\0a.Nathan Model\\experiments")
    for file in path.rglob("*_train_fix.log"):
        process_file(file)

if __name__ == "__main__":
    main()
