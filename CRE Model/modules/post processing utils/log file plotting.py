import json
import os
import re
import time
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
from scipy.stats import pearsonr, spearmanr




#main_path = 'D:\\A.Nathan\\1a.UWA24-Hons\\Honours Project\\0a.Code\\0a.Nathan Model\\logs'
main_path = 'D:\\A.Nathan\\1a.UWA24-Hons\\Honours Project\\0a.Code\\0a.Nathan Model\\0a.final analysis\\logs'
infile = 'log_best_final_long_20250503_082421.log'



def read_text_file(file_path):
    """Reads a text file and returns its content as a string, handling encoding issues."""
    with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
        return f.read()


def parse_log_file(filepath):
    data = {
        'step': [],
        'train_loss': [],
        'eval_loss': [],
        'span_f1': [],
        'rel_f1': [],
        'rel_mod_f1': [],
        'span_l_f1': [],
        'rel_l_f1': [],
        'rel_mod_l_f1': [],
        'test_loss': {},  # step → loss, optional
        'save_score': [],  #optional
    }

    with open(filepath, 'r', encoding='utf-8', errors='replace') as file:
        for line in file:
            # Extract test loss (optional)
            test_loss_match = re.search(r'step: (\d+), test loss mean: ([\d\.]+)', line)
            if test_loss_match:
                step = int(test_loss_match.group(1))
                test_loss = float(test_loss_match.group(2))
                data['test_loss'][step] = test_loss
                continue

            # Normal eval/training metrics
            if "train loss mean" in line:
                try:
                    step_match = re.search(r'step: (\d+)', line)
                    train_loss_match = re.search(r'train loss mean: (\d+\.\d+)', line)
                    eval_loss_match = re.search(r'eval loss mean: (\d+\.\d+)', line)
                    span_f1_match = re.search(r'Span: \(S: \d+, P: \d+\.\d+%, R: \d+\.\d+%, F1: (\d+\.\d+)%\)', line)
                    rel_f1_match = re.search(r'Rel: \(S: \d+, P: \d+\.\d+%, R: \d+\.\d+%, F1: (\d+\.\d+)%\)', line)
                    rel_mod_f1_match = re.search(r'Rel_mod: \(S: \d+, P: \d+\.\d+%, R: \d+\.\d+%, F1: (\d+\.\d+)%\)', line)
                    span_l_f1_match = re.search(r'Span_l: \(S: \d+, P: \d+\.\d+%, R: \d+\.\d+%, F1: (\d+\.\d+)%\)', line)
                    rel_l_f1_match = re.search(r'Rel_l: \(S: \d+, P: \d+\.\d+%, R: \d+\.\d+%, F1: (\d+\.\d+)%\)', line)
                    rel_mod_l_f1_match = re.search(r'Rel_mod_l: \(S: \d+, P: \d+\.\d+%, R: \d+\.\d+%, F1: (\d+\.\d+)%\)', line)
                    
                    if all([step_match, train_loss_match, eval_loss_match,
                            span_f1_match, rel_f1_match, rel_mod_f1_match,
                            span_l_f1_match, rel_l_f1_match, rel_mod_l_f1_match]):
                        data['step'].append(int(step_match.group(1)))
                        data['train_loss'].append(float(train_loss_match.group(1)))
                        data['eval_loss'].append(float(eval_loss_match.group(1)))
                        data['span_f1'].append(float(span_f1_match.group(1)))
                        data['rel_f1'].append(float(rel_f1_match.group(1)))
                        data['rel_mod_f1'].append(float(rel_mod_f1_match.group(1)))
                        data['span_l_f1'].append(float(span_l_f1_match.group(1)))
                        data['rel_l_f1'].append(float(rel_l_f1_match.group(1)))
                        data['rel_mod_l_f1'].append(float(rel_mod_l_f1_match.group(1)))
    
                        save_score_match = re.search(r'save_score: ([\d\.]+)', line)
                        data['save_score'].append(float(save_score_match.group(1)) if save_score_match else None)
    
                except Exception as e:
                    print("Error processing line: ", line)
                    print(e)
    return data




def plot_metrics(data):
    eval_loss_div_factor = 1
    plot_loose_matching = True
    plot_rel_mod = False

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    # Smoothing function
    def smooth_line(x, y, num=300):
        x_new = np.linspace(min(x), max(x), num)
        spl = make_interp_spline(x, y, k=2)  # Using a quadratic spline for smoother lines
        y_smooth = spl(x_new)
        return x_new, y_smooth

    # Plot training and evaluation loss with smoothing and dashed style
    train_x_smooth, train_y_smooth = smooth_line(data['step'], data['train_loss'])
    eval_x_smooth, eval_y_smooth = smooth_line(data['step'], [x / eval_loss_div_factor for x in data['eval_loss']])  # Scaling down eval loss for visualization
    ax1.set_xlabel('Step')
    ax1.set_ylabel(f'Train Loss (Eval Loss/{eval_loss_div_factor})' if eval_loss_div_factor != 1 else 'Loss')
    ax1.plot(train_x_smooth, train_y_smooth, label='Train Loss', color='r', linestyle='--', linewidth=2, alpha=0.5)
    ax1.plot(eval_x_smooth, eval_y_smooth, label='Eval Loss', color='b', linestyle='--', linewidth=2, alpha=0.5)

    # Optional test loss line — on same axis as eval loss
    if data.get('test_loss') and len(data['test_loss']) > 0:
        test_steps = sorted(data['test_loss'].keys())
        test_losses = [data['test_loss'][s] / eval_loss_div_factor for s in test_steps]  # apply same scaling
        test_x_smooth, test_y_smooth = smooth_line(test_steps, test_losses)
        ax1.plot(test_x_smooth, test_y_smooth, label='Test Loss', color='cyan', linestyle='--', linewidth=2, alpha=0.4)

    if data.get('save_score') and any(s is not None for s in data['save_score']):
        valid_save_scores = [(s, sc) for s, sc in zip(data['step'], data['save_score']) if sc is not None]
        save_steps, save_scores = zip(*valid_save_scores)
        save_x_smooth, save_y_smooth = smooth_line(save_steps, save_scores)
        ax1.plot(save_x_smooth, 10*save_y_smooth, label='Save Score', color='cyan', linestyle='--', linewidth=2, alpha=0.5)

    ax1.tick_params(axis='y', labelcolor='black')
    ax1.legend(loc='upper left')

    # Plot F1 Scores with smoothing
    span_f1_x_smooth, span_f1_y_smooth = smooth_line(data['step'], data['span_f1'])
    rel_f1_x_smooth, rel_f1_y_smooth = smooth_line(data['step'], data['rel_f1'])
    ax2.plot(span_f1_x_smooth, span_f1_y_smooth, label='Span F1', color='darkgreen', linewidth=2, alpha=0.5)
    ax2.plot(rel_f1_x_smooth, rel_f1_y_smooth, label='Rel F1', color='darkorange', linewidth=2, alpha=0.5)
    if plot_rel_mod:
        rel_mod_f1_x_smooth, rel_mod_f1_y_smooth = smooth_line(data['step'], data['rel_mod_f1'])
        ax2.plot(rel_mod_f1_x_smooth, rel_mod_f1_y_smooth, label='Rel_mod F1', color='darkorange', linewidth=6, alpha=0.2)
    if plot_loose_matching:
        span_l_f1_x_smooth, span_l_f1_y_smooth = smooth_line(data['step'], data['span_l_f1'])
        rel_l_f1_x_smooth, rel_l_f1_y_smooth = smooth_line(data['step'], data['rel_l_f1'])
        #rel_mod_l_f1_x_smooth, rel_mod_l_f1_y_smooth = smooth_line(data['step'], data['rel_mod_l_f1'])
        ax2.plot(span_l_f1_x_smooth, span_l_f1_y_smooth, label='Span F1 (loose)', color='lightgreen', linestyle='--', linewidth=1, alpha=0.5)
        ax2.plot(rel_l_f1_x_smooth, rel_l_f1_y_smooth, label='Rel F1 (loose)', color='orange', linestyle='--', linewidth=1, alpha=0.5)
        #ax2.plot(rel_mod_l_f1_x_smooth, rel_mod_l_f1_y_smooth, label='Rel_mod F1 (loose)', color='orange', linewidth=6, alpha=0.2)

    #setup the left y axis
    ax1.set_xlim(left=0)  # Start x-axis at 0
    max_step = max(data['step'])
    ax1.set_xticks(range(0, max_step + 1, 1000))  # Setting x ticks to have intervals of 1000
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray', alpha=0.7, axis='x')
    ax1.set_ylim(0)  # Start x-axis at 0
    #SEtup the right y axis
    ax2.set_ylabel('F1 Score')
    ax2.tick_params(axis='y', labelcolor='black')
    ax2.legend(loc='upper right')
    ax2.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray', alpha=0.7, axis='y')
    ax2.set_ylim(0)  # Start x-axis at 0

    # Set the title for the plot
    plt.title('Loss and F1 vs Training Step')

    fig.tight_layout()  # Adjust layout to make room for the legend

    plt.show()




def plot_scatter(data):
    # Compute correlation between test loss and eval loss (only for overlapping steps)
    if data.get('test_loss') and len(data['test_loss']) > 0:
        from scipy.stats import pearsonr, spearmanr

        common_steps = [s for s in data['step'] if s in data['test_loss']]
        if common_steps:
            eval_losses = [data['eval_loss'][data['step'].index(s)] for s in common_steps]
            test_losses = [data['test_loss'][s] for s in common_steps]

            pearson_corr, _ = pearsonr(eval_losses, test_losses)
            spearman_corr, _ = spearmanr(eval_losses, test_losses)

            print(f"\nTest vs Eval Loss Correlation (n={len(common_steps)}):")
            print(f"  Pearson:  {pearson_corr:.4f}")
            print(f"  Spearman: {spearman_corr:.4f}")

            # Optional: Scatterplot of test loss vs eval loss
            plt.figure(figsize=(6, 5))
            plt.scatter(eval_losses, test_losses, alpha=0.6, color='purple', edgecolor='black')
            plt.title('Test Loss vs Eval Loss')
            plt.xlabel('Eval Loss')
            plt.ylabel('Test Loss')
            plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.6)

            # Fit and plot regression line
            m, b = np.polyfit(eval_losses, test_losses, 1)
            x_line = np.linspace(min(eval_losses), max(eval_losses), 100)
            y_line = m * x_line + b
            # Add legend with regression equation
            plt.plot(x_line, y_line, color='gray', linestyle='--', linewidth=1.5)  # no label
            plt.legend()  # keeps legend for scatter only
            # Annotate correlation values
            plt.text(
                0.05, 0.95,
                f"Pearson r = {pearson_corr:.3f}\nSpearman ρ = {spearman_corr:.3f}",
                transform=plt.gca().transAxes,
                fontsize=10,
                verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray', alpha=0.8)
            )
            plt.tight_layout()
            plt.show()    




def main():
    # Path to the JSON file to read
    input_path = main_path + '\\' + infile

    data = parse_log_file(input_path)

    #plot_scatter(data)
    
    plot_metrics(data)






if __name__ == "__main__":
    main()