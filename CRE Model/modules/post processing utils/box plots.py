import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os



# Load CSV
input_dir = r'D:\A.Nathan\1a.UWA24-Hons\Honours Project\0a.Code\0a.Nathan Model\experiments'

file_name = 'log_metrics_raw_mod.csv'
#file_name = 'log_metrics_raw_val_mod.csv'
split = 'test'


file_path = os.path.join(input_dir, file_name)
df = pd.read_csv(file_path)

def plot_box(df, name_order, title):
    # Filter and order the DataFrame
    df_filtered = df[df['name'].isin(name_order)]
    df_filtered['name'] = pd.Categorical(df_filtered['name'], categories=name_order, ordered=True)
    
    # Melt for plotting
    df_melted = pd.melt(
        df_filtered,
        id_vars=['name'],
        value_vars=['Span_F1', 'Rel_F1'],
        var_name='Metric',
        value_name='F1_Score'
    )
    
    # Plot
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df_melted, x='name', y='F1_Score', hue='Metric')
    plt.title(title)
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.legend(title='')
    plt.show()

# 1) Brute Force vs Token Tagging (Span Width = 50)
plot_box(
    df,
    name_order=[
        'baseline span width 50',
        'brute force spans'
    ],
    title=f'Span Extraction: Brute Force vs Token Tagging ({split} set)'
)

# 2) Penalties
plot_box(
    df,
    name_order=[
        'baseline',
        'no lost rel penalties',
        'no consistency penalties',
        'no penalties'
    ],
    title=f'Effect of Penalties on Model Performance ({split} set)'
)

# 3) Graph
plot_box(
    df,
    name_order=[
        'no graph',
        'baseline',
        'graph-12heads, 6 layers',
        'graph-16heads, 8 layers',
        'graph-32heads, 16 layers'
    ],
    title=f'Graph Module Variants: Heads and Layers ({split} set)'
)

# 4) Relation Context Pooling
plot_box(
    df,
    name_order=[
        'baseline',
        'relation context crossattn-16heads',
        'relation context maxpool'
    ],
    title=f'Relation Context Pooling Method Comparison ({split} set)'
)
