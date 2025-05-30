import pandas as pd
import numpy as np
import os
from joblib import dump, load
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, recall_score, roc_auc_score, balanced_accuracy_score, roc_curve, matthews_corrcoef, precision_score, precision_recall_curve, auc, average_precision_score, RocCurveDisplay, PrecisionRecallDisplay
from sklearn.model_selection import cross_val_predict, StratifiedKFold, KFold, LeaveOneOut
import xgboost as xgb
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import shap
from sklearn.model_selection import GridSearchCV
import seaborn as sns
from sklearn.ensemble import StackingClassifier
from sklearn.base import clone

def remove_constant_string_des(df):
    #delete string value
    df = df.select_dtypes(exclude=['object'])
    #delete constant value
    for column in df.columns:
        if df[column].nunique() == 1:  # This checks if the column has only one unique value
            df = df.drop(column, axis=1)  # This drops the column from the DataFrame
    return df

def remove_highly_correlated_features(df, threshold=0.7):
    # Compute pairwise correlation of columns
    corr_matrix = df.corr().abs()
    # Create a mask for the upper triangle
    upper = corr_matrix.where(
        pd.DataFrame(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool), 
                     index=corr_matrix.index, columns=corr_matrix.columns)
    )
    # Identify columns to drop based on threshold
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    # Drop the columns from the DataFrame
    df_dropped = df.drop(columns=to_drop)
    return df_dropped


def y_prediction(model, x_train, y_train, col_name):
    y_pred = pd.DataFrame(model.predict(x_train), columns=[col_name]).set_index(x_train.index)
    acc = round(accuracy_score(y_train, y_pred),3)
    sen = round(recall_score(y_train, y_pred),3)
    mcc = round(matthews_corrcoef(y_train, y_pred),3)
    f1  = round(f1_score(y_train, y_pred),3)
    auc = round(roc_auc_score(y_train, y_pred),3)
    bcc = round(balanced_accuracy_score(y_train, y_pred),3)
    prc = round(average_precision_score(y_train, y_pred),3)
    # Calculate specificity
    tn, fp, fn, tp = confusion_matrix(y_train, y_pred).ravel()
    spc = round(tn / (tn + fp),3)

    # Create a DataFrame to store the metrics
    metrics = pd.DataFrame({
        'BACC': [bcc],
        'Accuracy': [acc],
        'Sensitivity': [sen],
        'Specificity': [spc],
        'MCC': [mcc],
        'AUROC': [auc],
        'AUPRC': [prc],
        'F1 Score': [f1],
    }, index=[col_name])
    return y_pred, metrics

def y_prediction_cv(model, x_train, y_train, col_name, output_dir='graph_metrics'):
    os.makedirs(output_dir, exist_ok=True)
    
    y_pred = cross_val_predict(model, x_train, y_train, cv=5)
    
    acc = round(accuracy_score(y_train, y_pred), 3)
    sen = round(recall_score(y_train, y_pred), 3)
    mcc = round(matthews_corrcoef(y_train, y_pred), 3)
    f1 = round(f1_score(y_train, y_pred), 3)
    auc = round(roc_auc_score(y_train, y_pred), 3)
    bcc = round(balanced_accuracy_score(y_train, y_pred), 3)
    prc = round(average_precision_score(y_train, y_pred), 3)
    
    tn, fp, fn, tp = confusion_matrix(y_train, y_pred).ravel()
    spc = round(tn / (tn + fp), 3)

    metrics = pd.DataFrame({
        'BACC': [bcc],
        'Accuracy': [acc],
        'Sensitivity': [sen],
        'Specificity': [spc],
        'MCC': [mcc],
        'AUROC': [auc],
        'AUPRC': [prc],
        'F1 Score': [f1],
    }, index=[col_name])
    return y_pred, metrics


def y_prediction_loocv(model, x_train, y_train, col_name, output_dir='graph_metrics'):
    os.makedirs(output_dir, exist_ok=True)

    loo = LeaveOneOut()
    preds = []
    probs = []
    indices = []

    for train_idx, val_idx in loo.split(x_train, y_train):
        x_tr, x_val = x_train.iloc[train_idx], x_train.iloc[val_idx]
        y_tr = y_train.iloc[train_idx]
        model.fit(x_tr, y_tr)
        y_val_pred = model.predict(x_val)
        y_val_proba = model.predict_proba(x_val)[:, 1]

        preds.append(y_val_pred[0])
        probs.append(y_val_proba[0])
        indices.append(x_val.index[0])

    # Collect predictions and probabilities
    y_pred_all = pd.DataFrame(preds, columns=[col_name], index=indices).sort_index()
    y_prob_all = pd.Series(probs, index=indices).sort_index()
    y_true_all = y_train.loc[y_pred_all.index]

    # Compute metrics on all predictions at once
    acc = round(accuracy_score(y_true_all, y_pred_all), 3)
    sen = round(recall_score(y_true_all, y_pred_all), 3)
    mcc = round(matthews_corrcoef(y_true_all, y_pred_all), 3)
    f1  = round(f1_score(y_true_all, y_pred_all), 3)
    auc = round(roc_auc_score(y_true_all, y_pred_all), 3)
    bcc = round(balanced_accuracy_score(y_true_all, y_pred_all), 3)
    prc = round(average_precision_score(y_true_all, y_pred_all), 3)
    tn, fp, fn, tp = confusion_matrix(y_true_all, y_pred_all).ravel()
    spc = round(tn / (tn + fp), 3)

    metrics = pd.DataFrame({
        'BACC': [bcc],
        'Accuracy': [acc],
        'Sensitivity': [sen],
        'Specificity': [spc],
        'MCC': [mcc],
        'AUROC': [auc],
        'AUPRC': [prc],
        'F1 Score': [f1],
    }, index=[col_name])

    return y_pred_all, metrics


def plot_auc_auprc_cv(model, x_train, y_train, col_name):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=None)
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    precisions = []
    auprcs = []
    mean_recall = np.linspace(0, 1, 100)

    # ROC Curve
    plt.figure(figsize=(5, 3))
    for i, (train_idx, val_idx) in enumerate(skf.split(x_train, y_train)):
        x_tr, x_val = x_train.iloc[train_idx], x_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        model.fit(x_tr, y_tr)
        if hasattr(model, "predict_proba"):
            y_score = model.predict_proba(x_val)[:, 1]
        else:
            y_score = model.decision_function(x_val)
        # ROC
        fpr, tpr, _ = roc_curve(y_val, y_score)
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        plt.plot(fpr, tpr, lw=1, alpha=0.7, label=f"Fold {i+1} (AUC = {roc_auc:.3f})")
        # Interpolate tpr
        tprs.append(np.interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
    # Plot mean ROC
    mean_tpr = np.mean(tprs, axis=0)
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    plt.plot(mean_fpr, mean_tpr, color='b', label=f"Mean AUROC = {mean_auc:.3f} ± {std_auc:.3f}", lw=2)
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', lw=1)
    plt.xlabel("1-Specificity", fontsize=12, fontstyle='italic', weight="bold")
    plt.ylabel("Sensitivity", fontsize=12, fontstyle='italic', weight="bold")
    plt.title(f"AUROC - {col_name}", fontsize=12, fontstyle='italic', weight="bold")
    plt.legend(bbox_to_anchor=(1,1), loc="upper left", fontsize='small')
    plt.tight_layout()
    plt.savefig(os.path.join("graph_metrics",f"{col_name}_roc_auc_cv.png"), dpi=500)
    plt.close()

    # Precision-Recall Curve
    plt.figure(figsize=(5, 3))
    for i, (train_idx, val_idx) in enumerate(skf.split(x_train, y_train)):
        x_tr, x_val = x_train.iloc[train_idx], x_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        model.fit(x_tr, y_tr)
        if hasattr(model, "predict_proba"):
            y_score = model.predict_proba(x_val)[:, 1]
        else:
            y_score = model.decision_function(x_val)
        precision, recall, _ = precision_recall_curve(y_val, y_score)
        auprc = average_precision_score(y_val, y_score)
        auprcs.append(auprc)
        plt.plot(recall, precision, lw=1, alpha=0.7, label=f"Fold {i+1} (AUPRC = {auprc:.3f})")
        # Interpolate precision
        precisions.append(np.interp(mean_recall, recall[::-1], precision[::-1]))
    # Plot mean PRC
    mean_precision = np.mean(precisions, axis=0)
    mean_auprc = np.mean(auprcs)
    std_auprc = np.std(auprcs)
    plt.plot(mean_recall, mean_precision, color='b', label=f"Mean AUPRC = {mean_auprc:.3f} ± {std_auprc:.3f}", lw=2)
    # Add [1,0] to [0,1] line
    plt.plot([1, 0], [0, 1], linestyle='--', color='gray', lw=1)
    plt.xlabel("Recall (Sensitivity)", fontsize=12, fontstyle='italic', weight="bold")
    plt.ylabel("Precision", fontsize=12, fontstyle='italic', weight="bold")
    plt.title(f"AUPRC - {col_name}", fontsize=12, fontstyle='italic', weight="bold")
    plt.legend(bbox_to_anchor=(1,1), loc="upper left", fontsize='small')
    plt.tight_layout()
    plt.savefig(os.path.join("graph_metrics",f"{col_name}_prc_auprc_cv.png"), dpi=500)
    plt.close()


def shap_plot(stacked_model, stack_test, name):
    explainer = shap.Explainer(stacked_model)
    shap_values = explainer(stack_test)
    shap.summary_plot(shap_values, stack_test, show=False, plot_type="bar", plot_size=(3, 5))
    plt.xlabel("mean|SHAP|", fontsize=12, fontstyle='italic',weight="bold")
    plt.savefig(name+'_shap.pdf', bbox_inches='tight')
    plt.close()


def nearest_neighbor_AD(x_train, name, k, z=3):
    from sklearn.neighbors import NearestNeighbors
    """
    Helper function to calculate the nearest neighbors and determine anomaly detection status.
    No need to call in the main function, as it is called within run_ad.
    """
    nn = NearestNeighbors(n_neighbors=k, algorithm='brute', metric='euclidean').fit(x_train)
    dump(nn, os.path.join(name, "ad_"+ str(k) +"_"+ str(z) +".joblib"))
    distance, index = nn.kneighbors(x_train)
    # Calculate mean and sd of distance in train set
    di = np.mean(distance, axis=1)
    # Find mean and sd of di
    dk = np.mean(di)
    sk = np.std(di)
    print('dk = ', dk)
    print('sk = ', sk)
    AD_status = ['within_AD' if di[i] < dk + (z * sk) else 'outside_AD' for i in range(len(di))]

    # Create DataFrame with index from x_test and the respective status
    df = pd.DataFrame(AD_status, index=x_train.index, columns=['AD_status'])
    return df, dk, sk

def run_ad(stacked_model, stack_train, y_train, name, z = 0.5):
    # Initialize lists to store metrics for plotting
    k_values = [3, 4, 5, 6, 7, 8, 9, 10]
    AUC_values = []
    PRC_values = []
    removed_compounds_values = []
    dk_values = []
    sk_values = []
    
    # Remove outside AD
    for i in k_values:
        print('k = ', i, 'z=', str(z))
        t, dk, sk = nearest_neighbor_AD(stack_train, name, i, z=z)
        t.to_csv("AD_train_set_"+str(i)+".csv")
        print(t['AD_status'].value_counts())
        # Remove outside AD Traiing set
        x_ad_train = stack_train[t['AD_status'] == 'within_AD']
        y_ad_train = y_train.loc[x_ad_train.index]
        y_pred_train = stacked_model.predict(x_ad_train)
        print("Check len of x_ad_train, y_ad_train, y_pred_ad_train: ", len(x_ad_train), len(y_ad_train), len(y_pred_train))
        # Evaluation
        print('Training set metrics:')
        auroc = roc_auc_score(y_ad_train, y_pred_train)
        auprc = round(average_precision_score(y_ad_train, y_pred_train), 3)
        print('AUROC: ', auroc,'AUPRC:', auprc)
        # Store metrics for plotting
        AUC_values.append(auroc)
        PRC_values.append(auprc)
        removed_compounds_values.append((t['AD_status'] == 'outside_AD').sum())
        dk_values.append(dk)
        sk_values.append(sk)
    k_values   = np.array(k_values)
    AUC_values = np.array(AUC_values)
    PRC_values = np.array(PRC_values)
    dk_values  = np.array(dk_values)
    sk_values  = np.array(sk_values)
    removed_compounds_values = np.array(removed_compounds_values)
    # Save table
    ad_metrics = pd.DataFrame({
        "k": k_values[:len(AUC_values)],  # Adjust if some values are skipped
        "AUROC": AUC_values,
        "AUPRC": PRC_values,
        "Removed Compounds": removed_compounds_values,
        "dk_values": dk_values,
        "sk_values": sk_values
    })
    ad_metrics = round(ad_metrics, 3)
    ad_metrics.to_csv("AD_metrics_"+name+"_"+ str(z)+ ".csv")
    

def y_random(stacked_features, y, metric_cv, metric_loocv, best_params, name):
    MCC_5cv = []
    MCC_loocv = []
    for i in range(1, 101):
        y_shuffled = pd.Series(y).sample(frac=1, replace=False, random_state=i).values
        model = xgb.XGBClassifier(**best_params).fit(stacked_features, y_shuffled)
        y_pred_cv, matrics_pred_cv = y_prediction_cv(model, stacked_features, pd.Series(y_shuffled, index=stacked_features.index), "Y-randomization_XGB_" + str(i))
        y_pred_loocv, matrics_pred_loocv = y_prediction_loocv(model, stacked_features, pd.Series(y_shuffled, index=stacked_features.index), "Y-randomization_XGB_" + str(i))
        MCC_5cv.append(matthews_corrcoef(y_shuffled, y_pred_cv))
        MCC_loocv.append(matthews_corrcoef(y_shuffled, y_pred_loocv))
    # Convert to DataFrame for easier handling and save the results
    MCC_5cv = pd.DataFrame(MCC_5cv, columns=['MCC']).set_index(pd.Index([f'Y-randomization_{i}' for i in range(1, 101)]))
    MCC_loocv = pd.DataFrame(MCC_loocv, columns=['MCC']).set_index(pd.Index([f'Y-randomization_{i}' for i in range(1, 101)]))
    MCC_5cv.to_csv("MCC_5cv_" + name + ".csv")
    MCC_loocv.to_csv("MCC_loocv_" + name + ".csv")
    size = [50]
    sizes = [20]
    # Use the correct row name for your stacking metrics
    x = [metric_cv.loc["Stacked_XGB", 'MCC']]
    y_ = [metric_loocv.loc["Stacked_XGB", 'MCC']]
    fig, ax = plt.subplots(figsize=(3, 3))
    ax.axvline(0.5, c='black', ls=':')
    ax.axhline(0.5, c='black', ls=':')
    ax.scatter(x, y_, s=size, c=['red'], marker='x', label='Our model')
    ax.scatter(MCC_5cv, MCC_loocv, c='blue', edgecolors='black', alpha=0.7, s=sizes, label='Y-randomization')
    ax.set_xlabel('$MCC_{5CV}$', fontsize=14, fontstyle='italic', weight='bold')
    ax.set_ylabel('$MCC_{LooCV}$', fontsize=14, fontstyle='italic', weight='bold')
    ax.legend(loc='lower right', fontsize='small')
    plt.tight_layout()
    plt.savefig("Y-randomization-" + name + "-classification.pdf", bbox_inches='tight')
    plt.close()

def run_ad_cv(stacked_model, stack_train, y_train, name, z=3):
    k_values = list(range(3, 11))
    results = []

    for k in k_values:
        print(f"===== AD 5-fold CV for k={k}, z={z} =====")
        # Save AD status for the full training set for this k
        ad_status_df, dk, sk = nearest_neighbor_AD(stack_train, name, k, z)
        ad_status_df.to_csv(f"AD_status_{name}_k{k}_z{z}.csv")
        # ...rest of your code...
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        fold_metrics = []
        removed_counts = []

        for fold, (train_idx, test_idx) in enumerate(skf.split(stack_train, y_train)):
            X_tr, X_te = stack_train.iloc[train_idx], stack_train.iloc[test_idx]
            y_tr, y_te = y_train.iloc[train_idx], y_train.iloc[test_idx]

            # Fit AD on training set
            _, dk, sk = nearest_neighbor_AD(X_tr, name, k, z)
            # Apply AD to test set
            from sklearn.neighbors import NearestNeighbors
            nn = NearestNeighbors(n_neighbors=k, algorithm='brute', metric='euclidean').fit(X_tr)
            test_dist, _ = nn.kneighbors(X_te)
            di_test = np.mean(test_dist, axis=1)
            test_ad_status = ['within_AD' if di_test[i] < dk + (z * sk) else 'outside_AD' for i in range(len(di_test))]
            test_ad_mask = np.array(test_ad_status) == 'within_AD'

            # Only evaluate on within-AD test samples
            X_te_ad = X_te.iloc[test_ad_mask]
            y_te_ad = y_te.iloc[test_ad_mask]
            removed_counts.append((~test_ad_mask).sum())

            if len(X_te_ad) == 0:
                continue  # skip if no within-AD samples

            y_pred = stacked_model.predict(X_te_ad)
            balanced_acc = round(balanced_accuracy_score(y_te_ad, y_pred), 3)
            mcc = round(matthews_corrcoef(y_te_ad, y_pred), 3)
            auroc = roc_auc_score(y_te_ad, y_pred)
            auprc = round(average_precision_score(y_te_ad, y_pred), 3)
            fold_metrics.append([balanced_acc, mcc, auroc, auprc])

        # Aggregate results for this k
        metrics_array = np.array(fold_metrics)
        metrics_mean = np.mean(metrics_array, axis=0)
        metrics_std = np.std(metrics_array, axis=0)
        results.append({
            "k": k,
            "BACC Mean": metrics_mean[0],
            "BACC Std": metrics_std[0],
            "MCC Mean": metrics_mean[1],
            "MCC Std": metrics_std[1],
            "AUROC Mean": metrics_mean[2],
            "AUROC Std": metrics_std[2],
            "AUPRC Mean": metrics_mean[3],
            "AUPRC Std": metrics_std[3],
            "Removed Compounds Mean": np.mean(removed_counts),
            "Removed Compounds Std": np.std(removed_counts),
        })
        print(f"k={k}: BACC={metrics_mean[0]:.3f}±{metrics_std[0]:.3f}, "
              f"MCC={metrics_mean[1]:.3f}±{metrics_std[1]:.3f}, "
              f"AUROC={metrics_mean[2]:.3f}±{metrics_std[2]:.3f}, "
              f"AUPRC={metrics_mean[3]:.3f}±{metrics_std[3]:.3f}, "
              f"Removed={np.mean(removed_counts):.1f}±{np.std(removed_counts):.1f}")

    # Save results
    ad_cv_metrics = pd.DataFrame(results)
    ad_cv_metrics.to_csv(f"AD_5CV_metrics_{name}_{z}.csv", index=False)
    print("Saved AD 5CV metrics table.")

def y_random_auroc_auprc(stacked_features, y, metric_cv, metric_loocv, best_params, name):
    AUROC_5cv = []
    AUPRC_5cv = []
    for i in range(1, 101):
        y_shuffled = pd.Series(y).sample(frac=1, replace=False, random_state=i).values
        model = xgb.XGBClassifier(**best_params).fit(stacked_features, y_shuffled)
        y_pred_cv = cross_val_predict(model, stacked_features, y_shuffled, cv=5, method="predict_proba")[:, 1]
        auroc = roc_auc_score(y_shuffled, y_pred_cv)
        auprc = average_precision_score(y_shuffled, y_pred_cv)
        AUROC_5cv.append(auroc)
        AUPRC_5cv.append(auprc)
    # Save results
    pd.DataFrame({'AUROC': AUROC_5cv, 'AUPRC': AUPRC_5cv}).to_csv(f"Yrandom_AUROC_AUPRC_{name}.csv", index=False)
    # Plot
    x = [metric_cv.loc["Stacked_XGB", 'AUROC']]
    y_ = [metric_cv.loc["Stacked_XGB", 'AUPRC']]
    fig, ax = plt.subplots(figsize=(3, 3))
    ax.axvline(0.5, c='black', ls=':')
    ax.axhline(0.5, c='black', ls=':')
    ax.scatter(x, y_, s=50, c=['red'], marker='x', label='Our model')
    ax.scatter(AUROC_5cv, AUPRC_5cv, c='blue', edgecolors='black', alpha=0.7, s=20, label='Y-randomization')
    ax.set_xlabel('$AUROC_{5CV}$', fontsize=14, fontstyle='italic', weight='bold')
    ax.set_ylabel('$AUPRC_{5CV}$', fontsize=14, fontstyle='italic', weight='bold')
    ax.legend(loc='lower right', fontsize='small')
    plt.tight_layout()
    plt.savefig("Y-randomization-" + name + "-AUROC-AUPRC.pdf", bbox_inches='tight')
    plt.close()


def main():
    all_results = []
    model_types = [
        ("RF", RandomForestClassifier(max_depth=3, max_features=5, random_state=None)),
        ("SVM", SVC(probability=True, kernel='rbf', C=1, gamma='scale', random_state=None)),
        ("XGB", xgb.XGBClassifier(max_depth=3, eval_metric='logloss', random_state=None))
    ]
    for x in ['AP2DC','AD2D','EState','CDKExt','CDK','CDKGraph','KRFPC','KRFP','MACCS','PubChem','SubFPC','SubFP', 'Combined']:
        print("#"*100)
        x_train = pd.read_csv(os.path.join('nafld', x+'.csv'), index_col=0)
        x_train_red = remove_constant_string_des(x_train)
        x_train_red = remove_highly_correlated_features(x_train_red, threshold=0.7)
        y_train = pd.read_csv(os.path.join('nafld', 'y_train.csv'), index_col=0)
        for model_prefix, model_instance in model_types:
            model = model_instance.fit(x_train_red, y_train.values.ravel())
            y_pred_cv, metrics_cv = y_prediction_cv(model, x_train_red, y_train, model_prefix + x)
            y_pred_loocv, metrics_loocv = y_prediction_loocv(model, x_train_red, y_train, model_prefix + x)
            plot_auc_auprc_cv(model, x_train_red, y_train, model_prefix +"_"+ x)

            # Collect metrics for result_cv.csv
            result_row_cv = {
                "Feature": x,
                "Model": model_prefix,
                "BACC": metrics_cv.loc[model_prefix + x, 'BACC'],
                "Sensitivity": metrics_cv.loc[model_prefix + x, 'Sensitivity'],
                "Specificity": metrics_cv.loc[model_prefix + x, 'Specificity'],
                "MCC": metrics_cv.loc[model_prefix + x, 'MCC'],
                "AUROC": metrics_cv.loc[model_prefix + x, 'AUROC'],
                "AUPRC": metrics_cv.loc[model_prefix + x, 'AUPRC'],
                "F1 Score": metrics_cv.loc[model_prefix + x, 'F1 Score'],
            }
            all_results.append(result_row_cv)

            # Save results after each fingerprint/model (append mode)
            results_cv = pd.DataFrame([result_row_cv])
            results_file = "results_cv.csv"
            if os.path.exists(results_file):
                existing = pd.read_csv(results_file)
                results_cv = pd.concat([existing, results_cv], ignore_index=True)
            results_cv.to_csv(results_file, index=False)
            print(f"✅ Results CV appended for {model_prefix} {x}")

            # Collect metrics for LOOCV
            result_row_loocv = {
                "Feature": x,
                "Model": model_prefix,
                "BACC": metrics_loocv.loc[model_prefix + x, 'BACC'],
                "Sensitivity": metrics_loocv.loc[model_prefix + x, 'Sensitivity'],
                "Specificity": metrics_loocv.loc[model_prefix + x, 'Specificity'],
                "MCC": metrics_loocv.loc[model_prefix + x, 'MCC'],
                "AUROC": metrics_loocv.loc[model_prefix + x, 'AUROC'],
                "AUPRC": metrics_loocv.loc[model_prefix + x, 'AUPRC'],
                "F1 Score": metrics_loocv.loc[model_prefix + x, 'F1 Score'],
            }
            results_loocv = pd.DataFrame([result_row_loocv])
            results_loocv_file = "results_loocv.csv"
            if os.path.exists(results_loocv_file):
                existing_loocv = pd.read_csv(results_loocv_file)
                results_loocv = pd.concat([existing_loocv, results_loocv], ignore_index=True)
            results_loocv.to_csv(results_loocv_file, index=False)
            print(f"✅ Results LOOCV appended for {model_prefix} {x}")

    # Stacked Generalization with OOF predictions
    # Define fingerprints and base models
    fingerprints = ['AP2DC','AD2D','EState','CDKExt','CDK','CDKGraph','KRFPC','KRFP','MACCS','PubChem','SubFPC','SubFP', 'Combined']
    base_model_defs = [
        ("RF", RandomForestClassifier(max_depth=3, max_features=5, random_state=None)),
        ("SVM", SVC(probability=True, kernel='rbf', C=1, gamma='scale', random_state=None)),
        ("XGB", xgb.XGBClassifier(max_depth=3, eval_metric='logloss', random_state=None))
    ]

    # Prepare data
    X_dict = {}
    for fp in fingerprints:
        X = pd.read_csv(os.path.join('nafld', f'{fp}.csv'), index_col=0)
        X = remove_constant_string_des(X)
        X = remove_highly_correlated_features(X, threshold=0.7)
        # save the processed fingerprint data
        X.to_csv(os.path.join('nafld', f'{fp}_reduced.csv'))
        X_dict[fp] = X
    y = pd.read_csv(os.path.join('nafld', 'y_train.csv'), index_col=0).values.ravel()

    # Generate OOF predictions for each base model
    from sklearn.model_selection import StratifiedKFold

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=None)
    stacked_features = pd.DataFrame(index=X_dict[fingerprints[0]].index)
    base_model_names = []

    for fp in fingerprints:
        X_fp = X_dict[fp]
        for model_name, model in base_model_defs:
            col_name = f"{model_name}_{fp}"
            base_model_names.append(col_name)
            oof_pred = np.zeros(len(X_fp))
            for train_idx, val_idx in skf.split(X_fp, y):
                X_tr, X_val = X_fp.iloc[train_idx], X_fp.iloc[val_idx]
                y_tr = y[train_idx]
                m = clone(model)
                m.fit(X_tr, y_tr)
                if hasattr(m, "predict_proba"):
                    oof_pred[val_idx] = m.predict_proba(X_val)[:, 1]
                else:
                    oof_pred[val_idx] = m.decision_function(X_val)
            stacked_features[col_name] = oof_pred

    # Train meta-model (XGBoost) on stacked features
    meta_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=None)
    meta_model.fit(stacked_features, y)

    # Evaluate meta-model
    y_pred_stack_cv, metrics_stack_cv = y_prediction_cv(meta_model, stacked_features, pd.Series(y, index=stacked_features.index), "Stacked_XGB")
    y_pred_stack_loocv, metrics_stack_loocv = y_prediction_loocv(meta_model, stacked_features, pd.Series(y, index=stacked_features.index), "Stacked_XGB")
    plot_auc_auprc_cv(meta_model, stacked_features, pd.Series(y, index=stacked_features.index), "Stacked_XGB")
    # Save the meta-model
    dump(meta_model, "meta_model_stacked_xgb.joblib")
    
    # Save the stacked features
    stacked_features.to_csv("stacked_features.csv")
    # Save the y_predictions from cross-validation and leave-one-out
    pd.DataFrame(y_pred_stack_cv).to_csv("y_pred_stack_cv.csv")
    pd.DataFrame(y_pred_stack_loocv).to_csv("y_pred_stack_loocv.csv")
    print("✅ Stacked XGB model trained and saved as meta_model_stacked_xgb.joblib")
    
    
    # Collect metrics for result_cv.csv
    result_row = {
        "Feature": "Stacked_All",  # or another appropriate label
        "Model": "Stacked_XGB",
        "BACC": metrics_stack_cv.loc["Stacked_XGB", 'BACC'],
        "Sensitivity": metrics_stack_cv.loc["Stacked_XGB", 'Sensitivity'],
        "Specificity": metrics_stack_cv.loc["Stacked_XGB", 'Specificity'],
        "MCC": metrics_stack_cv.loc["Stacked_XGB", 'MCC'],
        "AUROC": metrics_stack_cv.loc["Stacked_XGB", 'AUROC'],
        "AUPRC": metrics_stack_cv.loc["Stacked_XGB", 'AUPRC'],
        "F1 Score": metrics_stack_cv.loc["Stacked_XGB", 'F1 Score'],
    }
    # Save to results_cv.csv
    results_file = "results_cv.csv"
    results_cv = pd.DataFrame([result_row])
    if os.path.exists(results_file):
        existing = pd.read_csv(results_file)
        results_cv = pd.concat([existing, results_cv], ignore_index=True)
    results_cv.to_csv(results_file, index=False)


    result_row = {
        "Feature": "Stacked_All",  # or another appropriate label
        "Model": "Stacked_XGB",
        "BACC": metrics_stack_loocv.loc["Stacked_XGB", 'BACC'],
        "Sensitivity": metrics_stack_loocv.loc["Stacked_XGB", 'Sensitivity'],
        "Specificity": metrics_stack_loocv.loc["Stacked_XGB", 'Specificity'],
        "MCC": metrics_stack_loocv.loc["Stacked_XGB", 'MCC'],
        "AUROC": metrics_stack_loocv.loc["Stacked_XGB", 'AUROC'],
        "AUPRC": metrics_stack_loocv.loc["Stacked_XGB", 'AUPRC'],
        "F1 Score": metrics_stack_loocv.loc["Stacked_XGB", 'F1 Score'],
    }
    # Save to results_loocv.csv
    results_loocv_file = "results_loocv.csv"
    results_loocv = pd.DataFrame([result_row])
    if os.path.exists(results_loocv_file):
        existing_loocv = pd.read_csv(results_loocv_file)
        results_loocv = pd.concat([existing_loocv, results_loocv], ignore_index=True)
    results_loocv.to_csv(results_loocv_file, index=False)

    print("✅ Stacked XGB metrics saved to results_cv.csv and results_loocv.csv")

    # Only for tree-based models like XGBoost, LightGBM, RandomForest
    explainer = shap.TreeExplainer(meta_model)
    shap_values = explainer.shap_values(stacked_features)
    
    # Bar plot: mean absolute SHAP values per feature
    plt.figure(figsize=(6, 6))
    shap.summary_plot(shap_values, stacked_features, plot_type="bar", show=False)
    plt.xlabel("Mean |SHAP value|", fontsize=12, fontstyle='italic', weight='bold')
    plt.tight_layout()
    plt.savefig("shap_stacked_features_barplot.pdf", bbox_inches='tight')
    plt.close()
    
    # Beeswarm plot for detailed per-sample impact
    plt.figure(figsize=(6, 6))
    shap.summary_plot(shap_values, stacked_features, show=False)
    plt.tight_layout()
    plt.savefig("shap_stacked_features_beeswarm.pdf", bbox_inches='tight')
    plt.close()

    # Plot SHAP of SubFPC-XGB 
    
    subfpc_xgb = xgb.XGBClassifier(max_depth=3, eval_metric='logloss', random_state=None)
    subfpc_xgb.fit(X_dict['SubFPC'], y)
    explainer = shap.TreeExplainer(subfpc_xgb)
    shap_values = explainer.shap_values(X_dict['SubFPC'])

    # Plot SHAP of SubFP-XGB
    plt.figure(figsize=(6, 6))
    shap.summary_plot(shap_values, X_dict['SubFPC'] , show=False)
    plt.tight_layout()
    plt.savefig("shap_SubFPC_XGB_features_beeswarm.pdf", bbox_inches='tight')
    plt.close()
    
    # Plot SHAP of SubFP-XGB 
    
    subfpc_xgb = xgb.XGBClassifier(max_depth=3, eval_metric='logloss', random_state=None)
    subfpc_xgb.fit(X_dict['SubFP'], y)
    explainer = shap.TreeExplainer(subfpc_xgb)
    shap_values = explainer.shap_values(X_dict['SubFP'])

    # Plot SHAP of SubFP-XGB
    plt.figure(figsize=(6, 6))
    shap.summary_plot(shap_values, X_dict['SubFP'] , show=False)
    plt.tight_layout()
    plt.savefig("shap_SubFP_XGB_features_beeswarm.pdf", bbox_inches='tight')
    plt.close()

    # Y-randomization
    #y_random(
    #    stacked_features,
    #    y,
    #    metrics_stack_cv,
    #    metrics_stack_loocv,
    #    meta_model.get_params(),
    #    "Stacked_XGB"
    #)

    y_random_auroc_auprc(
        stacked_features,
        y,
        metrics_stack_cv,
        metrics_stack_loocv,
        meta_model.get_params(),
        "Stacked_XGB"
    )
    # AD
    #run_ad(meta_model, stacked_features, pd.Series(y, index=stacked_features.index), "Stacked_XGB", z=0.5)
    run_ad_cv(meta_model, stacked_features, pd.Series(y, index=stacked_features.index), "Stacked_XGB", z=3)
if __name__ == "__main__":
    main()

