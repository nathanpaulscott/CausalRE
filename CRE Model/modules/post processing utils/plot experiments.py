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


def prepare_df(df):
    # Prepare DataFrame
    df['score'] = df['Span_F1_mean'] * df['Rel_F1_mean']
    df = df.sort_values(by='score', ascending=False).reset_index(drop=True)
    df['rank'] = range(1, len(df) + 1)  # Assign rank after sorting by score

    # Sorting by std products
    df['sort_metric'] = df['Rel_F1_std'] * df['Span_F1_std']

    # Now sort by dataset first, then by rank
    df = df.sort_values(by=['dataset', 'rank'], ascending=[True, True]).reset_index(drop=True)

    return df




def plot_stuff(df):
    show_loose = True
    scale = 2   # 2sd is ~90% coverage

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
        #label = f"{row['rank']}"

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
    ax.set_xlabel("Relation F1 Mean (%)")
    ax.set_ylabel("Span F1 Mean (%)")
    ax.set_title("Span vs Relation F1")

    # Sorted legend
    handles, labels = ax.get_legend_handles_labels()
    # Sort handles and labels based on the extracted rank from labels
    sorted_handles_labels = sorted(zip(handles, labels), key=lambda x: int(x[1].split('.')[0]))
    sorted_handles, sorted_labels = zip(*sorted_handles_labels)  # Unzip sorted pairs
    ax.legend(sorted_handles, sorted_labels, title="Model (Ranked)")

    plt.show()



def plot_stuff_grid(df):
    show_loose = True
    scale = 2  # 2 standard deviations = ~90% coverage

    discriminator_vars = [
        'rank', 
        'dataset',
        'backbone',
        'shared backbone',
        'span filter type',
        'span construction',
        'rel construction',
        'rel context pooling',
        'top_k',
        'lstm/graph',
        'teacher forcing',
        'penalties'
    ]

    fig, axes = plt.subplots(3, 4, figsize=(22, 16))
    axes = axes.flatten()

    color_list = ['darkred', 'red', 'pink', 'orange', 'yellow', 'lightgreen', 'green', 'lightblue', 'darkblue', 'purple', 'black', 'brown', 'cyan', 'magenta']
    color_list = ['red', 'blue', 'green', 'yellow', 'orange', 'pink', 'lightgreen', 'darkblue', 'purple', 'black', 'brown', 'cyan', 'magenta','darkred']
    marker_list = ['o', 's', 'D', '^', 'P', 'X', '*']

    for idx, discriminator in enumerate(discriminator_vars):
        ax = axes[idx]
        unique_values = sorted(df[discriminator].dropna().unique())
        color_map = {val: color_list[i % len(color_list)] for i, val in enumerate(unique_values)}
        marker_map = {val: marker_list[i % len(marker_list)] for i, val in enumerate(unique_values)}

        handles = []

        for i, row in df.iterrows():
            value = row[discriminator]
            color = color_map.get(value, 'black')
            marker = marker_map.get(value, 'o')
            label = f"{row['rank']}. {row['base_name']}"

            if idx == 0:
                # Ellipse for uncertainty
                ellipse = mpatches.Ellipse((row['Rel_F1_mean'], row['Span_F1_mean']),
                                        width=row['Rel_F1_std'] * scale,
                                        height=row['Span_F1_std'] * scale,
                                        alpha=0.5, facecolor='none', edgecolor=color,
                                        linewidth=0.5, linestyle='--')
                ax.add_patch(ellipse)

            # Scatter point
            ax.scatter(row['Rel_F1_mean'], row['Span_F1_mean'],
                       #color=color, s=80, zorder=100-i,
                       #edgecolors='black', marker=marker, alpha=0.5)
                       facecolors='none', s=80, zorder=100-i,
                       edgecolors=color, marker=marker, alpha=0.5)

        # Set labels and grid
        #ax.set_xlim(0, 100)
        #ax.set_ylim(0, 100)
        #ax.set_xticks(np.arange(0, 101, 20))
        #ax.set_yticks(np.arange(0, 101, 20))
        
        ax.set_xlim(30, 80)
        ax.set_ylim(50, 100)
        ax.set_xticks(np.arange(30, 81, 10))  # Fix the ticks to match new x range
        ax.set_yticks(np.arange(50, 101, 10)) # Fix the ticks to match new y range
        
        ax.grid(True)
        ax.set_xlabel("Rel F1 Mean")
        ax.set_ylabel("Span F1 Mean")
        ax.set_title(discriminator, fontsize=11)

        # Now build legend for this subplot
        legend_elements = []
        for val in unique_values:
            legend_elements.append(
                mpatches.Patch(facecolor=color_map[val], label=str(val))
            )
        ax.legend(handles=legend_elements, title=discriminator, loc='best', fontsize='small', title_fontsize='small')

    # Turn off unused axes
    for j in range(len(discriminator_vars), len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()




def latex_table(df, cols):
    '''
    makes a latex table from teh given df
    '''
    # Create LaTeX table
    latex_table = df[cols].to_latex(index=False)

    # Save or print
    print(latex_table)




def main():
    # Assuming functions to collect and aggregate log metrics are defined elsewhere
    path = "D:\\A.Nathan\\1a.UWA24-Hons\\Honours Project\\0a.Code\\0a.Nathan Model\\0a.final analysis\\experiments"
    infile = 'experiments_final.csv'

    df = pd.read_csv(f'{path}\\{infile}')
    
    df = prepare_df(df)

    latex_table(df, ['rank', 
                     'dataset',
                     'backbone',
                     'shared backbone',
                     'span filter type',
                     'span construction',
                     'rel construction',
                     'rel context pooling',
                     'top_k',
                     'lstm/graph',
                     'teacher forcing',
                     'penalties',
                     ])

    #plot_stuff(df)
    plot_stuff_grid(df)




if __name__ == "__main__":
    main()