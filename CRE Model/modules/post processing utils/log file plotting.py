import json
import os
import re
import time
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline



# Adjustable parameters
#main_path = 'D:\\A.Nathan\\1a.UWA24-Hons\\Honours Project\\0a.Code\\0a.Nathan Model\\logs'
#infile = 'log_baseline_20250411_023839.log'
#infile = 'log_baseline_20250411_211810.log'
#infile = 'log_baseline_20250412_160925.log'
#infile = 'log_baseline_20250413_075212.log'


main_path = "D:\\A.Nathan\\1a.UWA24-Hons\\Honours Project\\0a.Code\\0a.Nathan Model\\experiments\\ttbe-spannathan-relmax-mixed_20250413_211003"
#infile = "log_ttbe-spannathan-relmax-mixed_33_train.log"
#infile = "log_ttbe-spannathan-relmax-mixed_129_train.log"
infile = "log_ttbe-spannathan-relmax-mixed_431_train.log"


#main_path = "D:\\A.Nathan\\1a.UWA24-Hons\\Honours Project\\0a.Code\\0a.Nathan Model\\experiments\\ttbe-spannathan-relmax-mixed_20250414_060141"
#infile = "log_ttbe-spannathan-relmax-mixed_33_train.log"
#infile = "log_ttbe-spannathan-relmax-mixed_129_train.log"
#infile = "log_ttbe-spannathan-relmax-mixed_431_train.log"


#main_path = "D:\\A.Nathan\\1a.UWA24-Hons\\Honours Project\\0a.Code\\0a.Nathan Model\\experiments\\ttBE-spanattn-relcrossattn-mixed_20250414_040825"
#infile = "log_ttBE-spanattn-relcrossattn-mixed_33_train.log"
#infile = "log_ttBE-spanattn-relcrossattn-mixed_129_train.log"
#infile = "log_ttBE-spanattn-relcrossattn-mixed_431_train.log"


#main_path = "D:\\A.Nathan\\1a.UWA24-Hons\\Honours Project\\0a.Code\\0a.Nathan Model\\experiments\\ttbeco-spannathan-relmax-mixed_20250413_235827"
#infile = "log_ttbeco-spannathan-relmax-mixed_33_train.log"
#infile = "log_ttbeco-spannathan-relmax-mixed_129_train.log"
#infile = "log_ttbeco-spannathan-relmax-mixed_431_train.log"



#main_path = "D:\\A.Nathan\\1a.UWA24-Hons\\Honours Project\\0a.Code\\0a.Nathan Model\\experiments\\ttbe-spannathan-relmax-mixed-temp-tf_20250414_025805"
#infile = "log_ttbe-spannathan-relmax-mixed-temp-tf_33_train.log"
#infile = "log_ttbe-spannathan-relmax-mixed-temp-tf_129_train.log"
#infile = "log_ttbe-spannathan-relmax-mixed-temp-tf_431_train.log"



#main_path = "D:\\A.Nathan\\1a.UWA24-Hons\\Honours Project\\0a.Code\\0a.Nathan Model\\experiments\\ttbe-spannathan-relmax-mixed-nolstm-nograph_20250414_044121"
#infile = "log_ttbe-spannathan-relmax-mixed-nolstm-nograph_33_train.log"
#infile = "log_ttbe-spannathan-relmax-mixed-nolstm-nograph_129_train.log"
#infile = "log_ttbe-spannathan-relmax-mixed-nolstm-nograph_431_train.log"



#main_path = "D:\\A.Nathan\\1a.UWA24-Hons\\Honours Project\\0a.Code\\0a.Nathan Model\\experiments\\ttbe-spannathan-relmax-mixed_topk-30-200_20250414_083319"
#infile = "log_ttbe-spannathan-relmax-mixed_topk-30-200_33_train.log"
#infile = "log_ttbe-spannathan-relmax-mixed_topk-30-200_129_train.log"
#infile = "log_ttbe-spannathan-relmax-mixed_topk-30-200_431_train.log"


#main_path = "D:\\A.Nathan\\1a.UWA24-Hons\\Honours Project\\0a.Code\\0a.Nathan Model\\experiments\\ttbe-spannathan-relmax-mixed_topk-30-50_20250414_103743"
#infile = "log_ttbe-spannathan-relmax-mixed_topk-30-50_33_train.log"
#infile = "log_ttbe-spannathan-relmax-mixed_topk-30-50_129_train.log"
#infile = "log_ttbe-spannathan-relmax-mixed_topk-30-50_431_train.log"



#main_path = "D:\\A.Nathan\\1a.UWA24-Hons\\Honours Project\\0a.Code\\0a.Nathan Model\\experiments\\ttBE-spannathan-relcrossattn-mixed_20250414_091823"
#infile = "log_ttBE-spannathan-relcrossattn-mixed_33_train.log"


main_path = "D:\\A.Nathan\\1a.UWA24-Hons\\Honours Project\\0a.Code\\0a.Nathan Model\\experiments\\ttbe-snat-rmax_h-t-conll04-bert_temp_tf_20250416_195709"
#infile = "log_ttbe-snat-rmax_h-t-conll04-bert_temp_tf_33_train.log"
#infile = "log_ttbe-snat-rmax_h-t-conll04-bert_temp_tf_129_train.log"
infile = "log_ttbe-snat-rmax_h-t-conll04-bert_temp_tf_431_train.log"


#main_path = "D:\\A.Nathan\\1a.UWA24-Hons\\Honours Project\\0a.Code\\0a.Nathan Model\\experiments\\ttbe-spannathan-relmax-mixed-bert-base-cased_20250414_134328"
#infile = "log_ttbe-spannathan-relmax-mixed-bert-base-cased_33_train.log"
#infile = "log_ttbe-spannathan-relmax-mixed-bert-base-cased_129_train.log"

main_path = "D:\\A.Nathan\\1a.UWA24-Hons\\Honours Project\\0a.Code\\0a.Nathan Model\\logs"
infile = "log_best_20250417_043537.log"


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
        'rel_mod_l_f1': []
    }

    with open(filepath, 'r', encoding='utf-8', errors='replace') as file:
        for line in file:
            if "train loss mean" in line:
                try:
                    step_match = re.search(r'step: (\d+)', line)
                    train_loss_match = re.search(r'train loss mean: (\d+\.\d+)', line)
                    eval_loss_match = re.search(r'eval loss mean: (\d+\.\d+)', line)
                    span_f1_match = re.search(r'Span: \(S: \d+, P: \d+\.\d+%, R: \d+\.\d+%, F1: (\d+\.\d+)%\)', line)
                    rel_f1_match = re.search(r'Rel: \(S: \d+, P: \d+\.\d+%, R: \d+\.\d+%, F1: (\d+\.\d+)%\)', line)
                    rel_mod_f1_match = re.search(r'Rel_mod: \(S: \d+, P: \d+\.\d+%, R: \d+\.\d+%, F1: (\d+\.\d+)%\)', line)
                    #loose matching results
                    span_l_f1_match = re.search(r'Span_l: \(S: \d+, P: \d+\.\d+%, R: \d+\.\d+%, F1: (\d+\.\d+)%\)', line)
                    rel_l_f1_match = re.search(r'Rel_l: \(S: \d+, P: \d+\.\d+%, R: \d+\.\d+%, F1: (\d+\.\d+)%\)', line)
                    rel_mod_l_f1_match = re.search(r'Rel_mod_l: \(S: \d+, P: \d+\.\d+%, R: \d+\.\d+%, F1: (\d+\.\d+)%\)', line)

                    if step_match and train_loss_match and eval_loss_match and \
                        span_f1_match and rel_f1_match and rel_mod_f1_match and \
                        span_l_f1_match and rel_l_f1_match and rel_mod_l_f1_match:
                        data['step'].append(int(step_match.group(1)))
                        data['train_loss'].append(float(train_loss_match.group(1)))
                        data['eval_loss'].append(float(eval_loss_match.group(1)))
                        data['span_f1'].append(float(span_f1_match.group(1)))
                        data['rel_f1'].append(float(rel_f1_match.group(1)))
                        data['rel_mod_f1'].append(float(rel_mod_f1_match.group(1)))
                        #loose matching results
                        data['span_l_f1'].append(float(span_l_f1_match.group(1)))
                        data['rel_l_f1'].append(float(rel_l_f1_match.group(1)))
                        data['rel_mod_l_f1'].append(float(rel_mod_l_f1_match.group(1)))
                    else:
                        print("Missing data in line: ", line)
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
    #SEtup the right y axis
    ax2.set_ylabel('F1 Score')
    ax2.tick_params(axis='y', labelcolor='black')
    ax2.legend(loc='upper right')
    ax2.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray', alpha=0.7, axis='y')

    # Set the title for the plot
    plt.title('Loss and F1 vs Training Step')

    fig.tight_layout()  # Adjust layout to make room for the legend
    plt.show()


def main():
    # Path to the JSON file to read
    input_path = main_path + '/' + infile

    data = parse_log_file(input_path)

    plot_metrics(data)






if __name__ == "__main__":
    main()