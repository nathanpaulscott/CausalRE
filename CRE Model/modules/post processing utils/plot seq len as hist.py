import json
import matplotlib.pyplot as plt
import numpy as np

def plot_seq_len_histograms(json_file_paths, labels):
    colors = ['blue', 'green', 'yellow', 'orange', 'red']  # Define a list of colors for the histograms
    plt.figure(figsize=(12, 8))

    # Collect all token lengths to determine bins and prepare data for stacking
    all_token_lengths = []
    token_lengths_by_file = []

    for json_file_path, limit in json_file_paths:
        with open(json_file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        token_lengths = [len(obs['tokens']) for i,obs in enumerate(data['data']) if i <= limit and len(obs['spans']) > 0 and len(obs['tokens']) > 10]
        token_lengths_by_file.append(token_lengths)
        all_token_lengths.extend(token_lengths)

    # Determine bins based on the overall data range
    global_min = min(all_token_lengths)
    global_max = max(all_token_lengths)
    bin_width = 5
    bins = np.arange(global_min, global_max + bin_width, bin_width)

    # Plot stacked histograms
    plt.hist(token_lengths_by_file, bins=bins, color=colors[:len(json_file_paths)], alpha=0.75, label=labels, stacked=True, edgecolor='black')

    plt.title('Histogram of Sequence Length Across Model Performance Class')
    plt.xlabel('Length of Sequence')
    plt.ylabel('Frequency')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.show()



# Example usage:
main_path = 'D:\\A.Nathan\\1a.UWA24-Hons\\Honours Project\\0a.Code\\0a.Nathan Model\\data\\unicausal'
infiles = [
    ('mixed_pred_final_Excellent', 1e6),
    ('mixed_pred_final_Good(done)', 1e6),
    ('mixed_pred_final_Ave(done)', 1e6),
    ('mixed_pred_final_Poor(done)', 1e6),
    ('mixed_pred_final_Terrible(131)', 131),
]
labels = ['Excellent', 'Good', 'Average', 'Poor', 'Terrible']

plot_seq_len_histograms([(f"{main_path}/{infile}.json", limit) for infile, limit in infiles], labels)