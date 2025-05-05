import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import random
import os
import re
from pathlib import Path
from matplotlib.transforms import Affine2D

def draw_radiating_dot(ax, x, y, color, radius=3, strength=0.5, resolution=50):
    # Create a radial decay mask
    size = resolution
    xx, yy = np.meshgrid(np.linspace(-1, 1, size), np.linspace(-1, 1, size))
    d = np.sqrt(xx**2 + yy**2)
    mask = np.clip(1 - d, 0, 1) ** 2  # smooth falloff

    # Create RGBA image using the mask as alpha
    img = np.zeros((size, size, 4))
    rgba = plt.matplotlib.colors.to_rgba(color)
    img[..., :3] = rgba[:3]
    img[..., 3] = strength * mask

    # Position the image centered on (x, y)
    trans = Affine2D().translate(x - radius, y - radius) + ax.transData
    ax.imshow(img, extent=(0, 2*radius, 0, 2*radius), transform=trans, zorder=2)


def draw_radiating_ellipse(ax, x, y, width, height, color, strength=0.5, resolution=50):
    # Create elliptical decay mask
    xx, yy = np.meshgrid(np.linspace(-1, 1, resolution), np.linspace(-1, 1, resolution))
    d = np.sqrt((xx / (width / max(width, height)))**2 + (yy / (height / max(width, height)))**2)
    mask = np.clip(1 - d, 0, 1) ** 2  # smooth falloff

    # Create RGBA image
    img = np.zeros((resolution, resolution, 4))
    rgba = plt.matplotlib.colors.to_rgba(color)
    img[..., :3] = rgba[:3]
    img[..., 3] = strength * mask

    # Position the image at correct data coordinates
    extent = (x - width, x + width, y - height, y + height)
    ax.imshow(img, extent=extent, origin='lower', zorder=1, interpolation='bilinear')



def extract_metrics_from_log(log_path):
    metrics = {
        'Span_P': None, 'Span_R': None, 'Span_F1': None,
        'Rel_P': None, 'Rel_R': None, 'Rel_F1': None,
        'Span_l_P': None, 'Span_l_R': None, 'Span_l_F1': None,
        'Rel_l_P': None, 'Rel_l_R': None, 'Rel_l_F1': None
    }

    with open(log_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        if 'Eval_type: final_val' in line:
            # Ensure there are enough lines after the current line
            if i + 10 < len(lines):
                eval_block = lines[i+2:i+10]
                break
    else:
        return None  # Skip if no evaluation block found

    pattern = re.compile(r'(\w+):\s+S:\s+\d+\s+P:\s+([\d.]+)%\s+R:\s+([\d.]+)%\s+F1:\s+([\d.]+)%')

    for line in eval_block:
        match = pattern.match(line.strip())
        if match:
            label, p, r, f1 = match.groups()
            if label in ['Span', 'Rel', 'Span_l', 'Rel_l']:
                metrics[f'{label}_P'] = float(p)
                metrics[f'{label}_R'] = float(r)
                metrics[f'{label}_F1'] = float(f1)

    return metrics



def collect_raw_log_metrics(root_dir):
    data = []
    root_path = Path(root_dir)
    for log_file in root_path.rglob('*_train.log'):
        metrics = extract_metrics_from_log(log_file)
        if metrics:
            log_name_full = log_file.stem.replace('_train', '')
            if "_" in log_name_full:
                base, seed = log_name_full.rsplit("_", 1)
                metrics['base_name'] = base
                metrics['seed'] = int(seed) if seed.isdigit() else seed
            else:
                metrics['base_name'] = log_name_full
                metrics['seed'] = None
            data.append(metrics)

    df = pd.DataFrame(data)
    df.to_csv(Path(root_dir) / 'log_metrics_raw.csv', index=False)
    print(f"✅ Raw metrics saved to: {Path(root_dir) / 'log_metrics_raw.csv'}")
    return df


def aggregate_log_metrics(df, root_dir):
    numeric_cols = [col for col in df.columns if col not in ['base_name', 'seed']]
    agg_df = df.groupby('base_name')[numeric_cols].agg(['mean', 'std']).reset_index()
    agg_df.columns = ['_'.join(col).strip('_') for col in agg_df.columns.values]
    agg_df.to_csv(Path(root_dir) / 'log_metrics_aggregated.csv', index=False)
    print(f"✅ Aggregated metrics saved to: {Path(root_dir) / 'log_metrics_aggregated.csv'}")
    return agg_df



def plot_stuff(agg_df):
    show_loose = True
    scale = 2   # 2sd is ~90% coverage

    # Prepare DataFrame
    df = agg_df.copy()
    df['score'] = df['Span_F1_mean'] * df['Rel_F1_mean']
    df = df.sort_values(by='score', ascending=False)
    df['rank'] = range(1, len(df) + 1)  # Assign rank

    # Sorting by std products
    df['sort_metric'] = df['Rel_F1_std'] * df['Span_F1_std']
    df.sort_values('sort_metric', ascending=False, inplace=True)

    # Set color and marker
    color_list = ['darkred', 'red', 'pink', 'orange', 'yellow', 'lightgreen', 'green', 'lightblue', 'darkblue', 'purple', 'black']
    colors = color_list * ((len(df) // len(color_list)) + 1)
    marker_list = ['o', 's', 'D', '^']

    fig, ax = plt.subplots(figsize=(10, 10))

    # Plotting
    for i, row in df.iterrows():
        color = colors[i % len(colors)]
        marker = marker_list[i % len(marker_list)]
        label = f"{row['rank']}. {row['base_name']}"

        # Main ellipse
        ellipse = mpatches.Ellipse((row['Rel_F1_mean'], row['Span_F1_mean']),
                                width=row['Rel_F1_std'] * scale,
                                height=row['Span_F1_std'] * scale,
                                alpha=0.5, facecolor='none', edgecolor=color,
                                linewidth=0.5, linestyle='--')
        ax.add_patch(ellipse)

        # Scatter points
        ax.scatter(row['Rel_F1_mean'], row['Span_F1_mean'], color=color, s=150, 
                zorder=100-i, label=label, alpha=0.5, edgecolors='black', marker=marker)

    # Set axis
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_xticks(np.arange(0, 101, 10))
    ax.set_yticks(np.arange(0, 101, 10))
    ax.grid(True)
    ax.set_xlabel("Rel F1 Mean (%)")
    ax.set_ylabel("Span F1 Mean (%)")
    ax.set_title("Span vs Rel F1 with SD Ellipses")

    # Sorted legend
    handles, labels = ax.get_legend_handles_labels()
    # Sort handles and labels based on the extracted rank from labels
    sorted_handles_labels = sorted(zip(handles, labels), key=lambda x: int(x[1].split('.')[0]))
    sorted_handles, sorted_labels = zip(*sorted_handles_labels)  # Unzip sorted pairs
    ax.legend(sorted_handles, sorted_labels, title="Model (Ranked)")

    plt.show()





def main():
    # Assuming functions to collect and aggregate log metrics are defined elsewhere
    path = "D:\\A.Nathan\\1a.UWA24-Hons\\Honours Project\\0a.Code\\0a.Nathan Model\\experiments"
    raw_df = collect_raw_log_metrics(path)
    agg_df = aggregate_log_metrics(raw_df, path)

    agg_df.to_csv(f'{path}\\experiments_processed.csv', index=False)

    plot_stuff(agg_df)



if __name__ == "__main__":
    main()