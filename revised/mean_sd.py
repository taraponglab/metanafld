import pandas as pd
from scipy.stats import shapiro, kruskal

# Make sure to install scikit-posthocs if not already:
# pip install scikit-posthocs

df = pd.read_csv("results_cv.csv")
grouped_stats = df.groupby(["Feature", "Model"]).agg(['mean', 'std'])

# Flatten column names
grouped_stats.columns = ['_'.join(col).strip() for col in grouped_stats.columns.values]
grouped_stats.reset_index(inplace=True)
grouped_stats = grouped_stats.round(3)

metrics = ["BACC", "Sensitivity", "Specificity", "MCC", "AUROC", "AUPRC", "F1 Score"]

print("\nShapiro-Wilk Test for Normality (per Model):")
for metric in metrics:
    for model in df["Model"].unique():
        stat, p = shapiro(df[df["Model"] == model][metric])
        print(f"{metric} ({model}): W={stat:.4f}, p-value={p:.4g}")

print("\nKruskal-Wallis Test across Features (per Model):")
for metric in metrics:
    for model in df["Model"].unique():
        groups = [group[metric].values for name, group in df[df["Model"] == model].groupby("Feature")]
        # Only keep groups with more than 0 samples
        groups = [g for g in groups if len(g) > 0]
        if len(groups) > 1:
            stat, p = kruskal(*groups)
            print(f"{metric} ({model}): H={stat:.4f}, p-value={p:.4g}")
        else:
            print(f"{metric} ({model}): Not enough groups for Kruskal-Wallis test.")

# Post hoc Dunn's Test with Bonferroni correction (per Model)
import scikit_posthocs as sp

print("\nPost hoc Dunn's Test with Bonferroni correction (per Model):")
for metric in metrics:
    for model in df["Model"].unique():
        model_df = df[df["Model"] == model]
        n_groups = model_df["Feature"].nunique()
        if n_groups > 1:
            dunn_result = sp.posthoc_dunn(model_df, val_col=metric, group_col='Feature', p_adjust='bonferroni')
            print(f"\n{metric} ({model}):\n", dunn_result)
            dunn_result.to_csv(f"dunn_5CV_{metric}_{model}_reduced.csv")
        else:
            print(f"{metric} ({model}): Not enough groups for Dunn's test.")

# Combine mean ± std for export
combined_df = pd.DataFrame()
combined_df["Feature"] = grouped_stats["Feature"]
combined_df["Model"] = grouped_stats["Model"]

for metric in metrics:
    combined_df[metric] = grouped_stats[f"{metric}_mean"].round(3).astype(str) + " ± " + grouped_stats[f"{metric}_std"].round(3).astype(str).replace("\u00ac", "")
combined_df.to_excel("grouped_5CV_stats_reduced.xlsx", index=False)
print(combined_df)


# Repeat for test set
df = pd.read_csv("results_loocv.csv")
grouped_stats = df.groupby(["Feature", "Model"]).agg(['mean', 'std'])
# Flatten column names
grouped_stats.columns = ['_'.join(col).strip() for col in grouped_stats.columns.values]
grouped_stats.reset_index(inplace=True)
grouped_stats = grouped_stats.round(3)

print("\nShapiro-Wilk Test for Normality (per Model):")
for metric in metrics:
    for model in df["Model"].unique():
        stat, p = shapiro(df[df["Model"] == model][metric])
        print(f"{metric} ({model}): W={stat:.4f}, p-value={p:.4g}")

print("\nKruskal-Wallis Test across Features (per Model):")
for metric in metrics:
    for model in df["Model"].unique():
        groups = [group[metric].values for name, group in df[df["Model"] == model].groupby("Feature")]
        # Only keep groups with more than 0 samples
        groups = [g for g in groups if len(g) > 0]
        if len(groups) > 1:
            stat, p = kruskal(*groups)
            print(f"{metric} ({model}): H={stat:.4f}, p-value={p:.4g}")
        else:
            print(f"{metric} ({model}): Not enough groups for Kruskal-Wallis test.")

# Post hoc Dunn's Test with Bonferroni correction (per Model)
print("\nPost hoc Dunn's Test with Bonferroni correction (per Model):")
for metric in metrics:
    for model in df["Model"].unique():
        model_df = df[df["Model"] == model]
        n_groups = model_df["Feature"].nunique()
        if n_groups > 1:
            dunn_result = sp.posthoc_dunn(model_df, val_col=metric, group_col='Feature', p_adjust='bonferroni')
            print(f"\n{metric} ({model}):\n", dunn_result)
            dunn_result.to_csv(f"dunn_test_{metric}_{model}_reduced.csv")
        else:
            print(f"{metric} ({model}): Not enough groups for Dunn's test.")

# Combine mean ± std for export
combined_df = pd.DataFrame()
combined_df["Feature"] = grouped_stats["Feature"]
combined_df["Model"] = grouped_stats["Model"]

for metric in metrics:
    combined_df[metric] = grouped_stats[f"{metric}_mean"].round(3).astype(str) + " ± " + grouped_stats[f"{metric}_std"].round(3).astype(str).replace("\u00ac", "")
combined_df.to_excel("grouped_loocv_stats_reduced.xlsx", index=False)
print(combined_df)
