import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
from matplotlib.ticker import MaxNLocator


def parse_log(file_path):
    """
    Parse lines like:
      … loss breakdown: step: 99.13, tagger_loss: 0.55, …
    into a DataFrame of step + losses.
    """
    # 1) allow decimals in the 'step' capture
    pattern = re.compile(r"loss breakdown:\s*step:\s*([\d\.]+),\s*(.+)")
    records = []
    with open(file_path, 'r') as f:
        for line in f:
            m = pattern.search(line)
            if not m:
                continue
            # 2) cast step to float
            step = float(m.group(1))
            kv_str = m.group(2)
            parts = [p.strip() for p in kv_str.split(',')]
            entry = {'step': step}
            for part in parts:
                k, v = part.split(':', 1)
                entry[k.strip()] = float(v)
            records.append(entry)

    df = pd.DataFrame(records)
    return df.sort_values('step').reset_index(drop=True)



def smooth_line(x, y, num=300):
    """
    Quadratic‐spline smoothing of (x,y).

    Args:
        x (1d array): original x values
        y (1d array): original y values
        num (int): number of points in the smoothed curve

    Returns:
        x_new, y_smooth: the new, evenly spaced x values and their smoothed y’s
    """
    x_new = np.linspace(min(x), max(x), num)
    spl = make_interp_spline(x, y, k=3)  # quadratic spline
    y_smooth = spl(x_new)
    return x_new, y_smooth


if __name__ == "__main__":
    global_ymin = 0
    global_ymax = 1
    num=1000
    plot_cols = [
        'tagger_loss', 
        'filter_loss_span', 
        'filter_loss_rel', 
        'lost_rel_loss', 
        'filter_loss_graph', 
        'prune_loss', 
        'pred_loss_span', 
        'pred_loss_rel',
        'RawTotNorm'
    ]

    #main_path = 'D:\\A.Nathan\\1a.UWA24-Hons\\Honours Project\\0a.Code\\0a.Nathan Model\\0a.final analysis\\detailed train logs'
    #main_path = 'D:\\A.Nathan\\1a.UWA24-Hons\\Honours Project\\0a.Code\\0a.Nathan Model\\logs'
    main_path = 'D:\\A.Nathan\\1a.UWA24-Hons\\Honours Project\\0a.Code\\0a.Nathan Model\\0a.final analysis\\logs'
    filename = 'log_best_final_long_20250503_082421.log'
    
    df = parse_log(main_path + '\\' + filename)
    
    # Print the parsed DataFrame
    print(df.head())

    # Determine loss columns (exclude 'step')
    loss_cols = [c for c in df.columns if c != 'step']
    loss_cols = [x for x in plot_cols if x in loss_cols]
    
    #global_ymin = df[loss_cols].min().min()
    #global_ymax = df[loss_cols].max().max()

    # Create a vertical subplot for each loss
    n = len(loss_cols)
    fig, axes = plt.subplots(n, 1, figsize=(10, 2.5 * n), sharex=True)

    steps = df['step'].values
    for ax, col in zip(axes, loss_cols):
        ax.plot(steps, df[col], label=col, linestyle = '-', linewidth=1, alpha=0.2)
        x_smooth, y_smooth = smooth_line(steps, df[col].values, num=num)
        ax.plot(x_smooth, y_smooth, label=col, linestyle = '-', linewidth=2, alpha=0.9)
        ax.set_ylabel(col)
        ax.set_ylim(global_ymin, global_ymax)  # unify y-axis range
        ax.grid(True)


    # Configure shared x-axis on bottom subplot
    axes[-1].set_xlabel('Step')
    axes[-1].xaxis.set_major_locator(MaxNLocator(nbins=10, integer=True))
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()