import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.lines import Line2D



def parse_raw_tot_norm(file_path):
    """
    Extract 'train step', 'RawTotNorm', 'TotNorm', and 'TotNorm.SD' from log lines.
    """
    pattern = re.compile(r"train step:\s*(\d+).*RawTotNorm:\s*([\d\.]+)\s*\| TotNorm:\s*([\d\.]+)\s*\| TotNorm.SD:\s*([\d\.]+)")
    steps, raw_norms, tot_norms, tot_norms_sd = [], [], [], []
    with open(file_path, 'r') as f:
        for line in f:
            m = pattern.search(line)
            if m:
                steps.append(int(m.group(1)))
                raw_norms.append(float(m.group(2)))
                tot_norms.append(float(m.group(3)))
                tot_norms_sd.append(float(m.group(4)))
    return np.array(steps), np.array(raw_norms), np.array(tot_norms), np.array(tot_norms_sd)


if __name__ == "__main__":
    #main_path = 'D:\\A.Nathan\\1a.UWA24-Hons\\Honours Project\\0a.Code\\0a.Nathan Model\\0a.final analysis\\logs'
    main_path = 'D:\\A.Nathan\\1a.UWA24-Hons\\Honours Project\\0a.Code\\0a.Nathan Model\\logs'
    filename = 'log_best_final_long2_20250501_142001.log'



    steps, raw_norms, tot_norms, tot_norms_sd = parse_raw_tot_norm(main_path + '\\' + filename)

    fig, ax1 = plt.subplots(figsize=(10, 4))

    ax1.set_xlabel('Training Step')
    ax1.set_ylabel('Norms', color='tab:blue')

    # Thicker lines + alpha
    ax1.plot(steps, raw_norms, color='tab:blue', linewidth=0.5, alpha=0.5)
    ax1.plot(steps, tot_norms, color='tab:green', linewidth=0.5, linestyle='--', alpha=0.5)
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.xaxis.set_major_locator(MaxNLocator(nbins=10, integer=True))
    #ax1.set_xlim(8000, 9000)
    ax1.set_ylim(0, 500)
    
    #ax2 = ax1.twinx()
    #ax2.set_ylabel('TotNorm.SD', color='tab:red')
    #ax2.plot(steps, tot_norms_sd, color='tab:red', linewidth=0.25, linestyle=':', alpha=0.5)
    #ax2.tick_params(axis='y', labelcolor='tab:red')

    fig.suptitle('RawTotNorm, TotNorm and TotNorm.SD over Training')

    # Clean legend without alpha
    custom_lines = [
        Line2D([0], [0], color='tab:blue', lw=2.0, linestyle='-'),
        Line2D([0], [0], color='tab:green', lw=2.0, linestyle='--'),
        Line2D([0], [0], color='tab:red', lw=2.0, linestyle=':')
    ]
    fig.legend(custom_lines, ['RawTotNorm', 'TotNorm', 'TotNorm.SD'], loc='upper right')

    fig.tight_layout()
    plt.grid(True)
    plt.show()
