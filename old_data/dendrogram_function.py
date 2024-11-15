import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as sch
import matplotlib.cm as cm
import radialtree as rt  # Assuming radialtree is the correct package for your radialTreee function

def plot_circular_dendrogram(df, class_col='CLASS', pathway_col='PATHWAY', source_col='SOURCE', chemical_col='CHEMICAL_NAME'):
    # Encode the CLASS, PATHWAY, and Source columns for clustering
    df_encoded = pd.get_dummies(df[[class_col, pathway_col, source_col]])
    Y = sch.linkage(df_encoded, method="average")
    Z2 = sch.dendrogram(
        Y,
        labels=df[chemical_col].values,
        no_plot=True,
    )

    # Prepare the figure
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.set_aspect(1)

    # Define specific colors for each class and source
    class_colors = {
        'Inducer'  : '#FF6347',  # Tomato red
        'Inhibitor': '#4682B4',  # Steel blue
    }
    source_colors = {
        'Drugs' : '#9e0142',
        'Natural products' : '#d53e4f',
        'Chemicals' : '#f46d43',
        'Metabolites' : '#fdae61',
        'Fatty acids' : '#fee08b',
        'Amino acids' : '#ffffbf',
        'Sugars' : '#e6f598',
        'Hormones' : '#abdda4',
        'Polysaccharides' : '#66c2a5',
        'Peptides' : '#3288bd',
        'Vitamins' : '#5e4fa2'
    }
    colors_legends= {"Classes": {"colors": ['#FF6347','#4682B4'], "labels": ['Inducer', 'Inhibitor']},
                     "Source": {"colors": list(source_colors.values()), "labels": list(source_colors.keys())}}
    # Create color arrays for leaves
    leaf_colors = [class_colors.get(cls, '#FFFF00') for cls in df[class_col]]
    leaf_source_colors = [source_colors.get(src, '#FFFF00') for src in df[source_col]]
    print(leaf_source_colors)
    # Plot the radial tree
    rt.radialTreee(Z2, ax=ax, pallete="Dark2", fontsize=10, colorlabels={"Classes": leaf_colors, "Source": leaf_source_colors}, colorlabels_legend=colors_legends)
    for line in ax.lines:
        line.set_linewidth(1)  # Adjust the line width
    plt.tight_layout()
    plt.savefig("dendrogram.png", dpi=500)
    plt.show()

# Example usage
df = pd.read_csv("dataset.csv")  # Load your data here
plot_circular_dendrogram(df)
