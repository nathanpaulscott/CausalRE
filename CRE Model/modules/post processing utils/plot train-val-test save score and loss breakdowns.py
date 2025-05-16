
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
from scipy.stats import pearsonr
from matplotlib.ticker import MaxNLocator

#LOG_PATH = 'D:\\A.Nathan\\1a.UWA24-Hons\\Honours Project\\0a.Code\\0a.Nathan Model\\0a.final analysis\\logs\\full loss temp TF.log' 
#LOG_PATH = 'D:\\A.Nathan\\1a.UWA24-Hons\\Honours Project\\0a.Code\\0a.Nathan Model\\0a.final analysis\\logs\\full loss always TF.log' 
LOG_PATH = 'D:\\A.Nathan\\1a.UWA24-Hons\\Honours Project\\0a.Code\\0a.Nathan Model\\logs\\log_test_20250511_164338.log' 

#LOG_PATH = 'D:\\A.Nathan\\1a.UWA24-Hons\\Honours Project\\0a.Code\\0a.Nathan Model\\logs\\log_test2_20250509_141706.log'

def smooth_line(x, y, num=300):
    x_new = np.linspace(min(x), max(x), num)
    spl = make_interp_spline(x, y, k=3)
    return x_new, spl(x_new)

def extract_save_scores(log_path):
    val_data, test_data = {}, {}
    with open(log_path, 'r', encoding='utf-8') as f:
        for line in f:
            if "eval loss mean" in line:
                step = int(re.search(r'step: (\d+)', line).group(1))
                s = [float(x) for x in re.findall(r'Span: \(S: \d+, P: ([\d.]+)%, R: ([\d.]+)%, F1: ([\d.]+)%\)', line)[0]]
                r = [float(x) for x in re.findall(r'Rel: \(S: \d+, P: ([\d.]+)%, R: ([\d.]+)%, F1: ([\d.]+)%\)', line)[0]]

                s_balance = (min(s[0], s[1]) / max(s[0], s[1]))**2 if max(s[0], s[1]) > 0 else 0
                r_balance = (min(r[0], r[1]) / max(r[0], r[1]))**2 if max(r[0], r[1]) > 0 else 0
                val_data[step] = 0.5 * (s[2] * s_balance + r[2] * r_balance)

            if "test loss mean" in line:
                step = int(re.search(r'step: (\d+)', line).group(1))
                s = [float(x) for x in re.findall(r'Span: \(S: \d+, P: ([\d.]+)%, R: ([\d.]+)%, F1: ([\d.]+)%\)', line)[0]]
                r = [float(x) for x in re.findall(r'Rel: \(S: \d+, P: ([\d.]+)%, R: ([\d.]+)%, F1: ([\d.]+)%\)', line)[0]]

                s_balance = (min(s[0], s[1]) / max(s[0], s[1]))**2 if max(s[0], s[1]) > 0 else 0
                r_balance = (min(r[0], r[1]) / max(r[0], r[1]))**2 if max(r[0], r[1]) > 0 else 0
                test_data[step] = 0.5 * (s[2] * s_balance + r[2] * r_balance)

    return pd.DataFrame({'step': list(val_data.keys()), 'val': list(val_data.values())}), \
           pd.DataFrame({'step': list(test_data.keys()), 'test': list(test_data.values())})


def plot_save_scores(val_df, test_df):
    plt.figure(figsize=(10, 5))
    x_v, y_v = smooth_line(val_df['step'], val_df['val'])
    x_t, y_t = smooth_line(test_df['step'], test_df['test'])
    plt.plot(x_v, y_v, color='green', label='Val')
    plt.plot(x_t, y_t, color='red', label='Test')
    plt.legend(), plt.grid(True), plt.xlabel("Step"), plt.ylabel("Save Score"), plt.title("Save Score Progress")
    plt.tight_layout(), plt.show()

    merged = pd.merge(val_df, test_df, on='step')
    r, _ = pearsonr(merged['val'], merged['test'])
    plt.scatter(merged['val'], merged['test'], alpha=0.6, color='purple')
    plt.title(f"Test vs Val Save Score\nPearson r = {r:.3f}")
    plt.xlabel("Val"), plt.ylabel("Test")
    plt.grid(True)
    plt.tight_layout(), plt.show()


def parse_loss_breakdown(log_path, key):
    records = []
    loss_keys_order = []

    # Corrected prefix values — NO trailing colon
    prefix = {
        'train': 'loss breakdown',
        'eval': 'eval loss breakdown',
        'test': 'test loss breakdown'
    }[key]

    # Pattern correctly includes the colon after the prefix
    pattern = re.compile(rf"^\S+\s+\S+\s+-\s+INFO\s+-\s+root\s+-\s+{re.escape(prefix)}:\s*step:\s*([\d.]+),\s*(.+)")

    with open(log_path, 'r', encoding='utf-8') as f:
        for line in f:
            m = pattern.search(line)
            if m:
                step = float(m.group(1))
                kv_str = m.group(2)
                parts = [p.strip() for p in kv_str.split(',')]
                if not loss_keys_order:  # First line — store key order
                    loss_keys_order = [p.split(':')[0].strip() for p in parts]
                entry = {'step': step}
                for part in parts:
                    k, v = part.split(':')
                    entry[k.strip()] = float(v)
                records.append(entry)

    df = pd.DataFrame(records).sort_values('step').reset_index(drop=True)
    return df, loss_keys_order



def plot_loss(all_dfs: dict, loss_keys_order: list, title: str, colors: dict):
    fig, axes = plt.subplots(len(loss_keys_order), 1, figsize=(10, 3 * len(loss_keys_order)), sharex=True)

    label_map = {
        'train': 'Train Loss', 
        'eval': 'Val Loss', 
        'test': 'Test Loss'
    }
    y_threshold = 1.0  # if max across all is > this, set y-lim to 5

    for i, key in enumerate(loss_keys_order):
        ax = axes[i]
        max_val = max(df[key].max() for df in all_dfs.values() if key in df)

        for label, df in all_dfs.items():
            if key in df.columns:
                x = df['step'].values
                y = df[key].values
                ax.plot(x, y, color=colors[label], alpha=0.2)
                x_smooth, y_smooth = smooth_line(x, y)
                ax.plot(x_smooth, y_smooth, color=colors[label], linewidth=2, label=label_map[label] if i == 0 else None)

        ax.set_ylabel(key)
        ax.set_ylim(0, 5 if max_val > y_threshold else 1)
        ax.grid(True)

    axes[-1].set_xlabel("Step")
    axes[0].set_title(title)
    axes[0].legend(loc='upper right')
    plt.tight_layout(rect=[0, 0, 1, 0.97])  # reserve space at top for title
    plt.show()



def main():
    val_df, test_df = extract_save_scores(LOG_PATH)
    plot_save_scores(val_df, test_df)

    train_df, loss_order = parse_loss_breakdown(LOG_PATH, 'train')
    eval_df, _ = parse_loss_breakdown(LOG_PATH, 'eval')
    test_df, _ = parse_loss_breakdown(LOG_PATH, 'test')

    plot_loss(
        all_dfs={'train': train_df, 'eval': eval_df, 'test': test_df},
        loss_keys_order=loss_order,
        title="Train / Eval / Test Loss Breakdown",
        colors={'train': 'blue', 'eval': 'green', 'test': 'red'}
    )

def main_no_test():
    #val_df, test_df = extract_save_scores(LOG_PATH)
    #plot_save_scores(val_df, test_df)

    train_df, loss_order = parse_loss_breakdown(LOG_PATH, 'train')
    eval_df, _ = parse_loss_breakdown(LOG_PATH, 'eval')

    plot_loss(
        all_dfs={'train': train_df, 'eval': eval_df},
        loss_keys_order=loss_order,
        title="Train / Eval Loss Breakdown",
        colors={'train': 'blue', 'eval': 'green'}
    )



if __name__ == '__main__':
    #main()
    main_no_test()
