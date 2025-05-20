import pandas as pd
import numpy as np
import os
from joblib import dump, load
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, recall_score, roc_auc_score, balanced_accuracy_score, roc_curve, matthews_corrcoef, precision_score, precision_recall_curve, auc, average_precision_score
from sklearn.model_selection import cross_val_predict, StratifiedKFold, KFold, LeaveOneOut
import xgboost as xgb
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import shap
from sklearn.model_selection import GridSearchCV
import seaborn as sns

""" 
1. Load x and y
2. Train stacked model (BL + Stack + 5-CV) (save)
3. Evaluate performance of all train, cv, test (save)
4. SHAP (save)
5. Y-random (save)
6. AD (save)
"""


def y_prediction(model, x_train, y_train, col_name):
    y_pred = pd.DataFrame(model.predict(x_train), columns=[col_name]).set_index(x_train.index)
    acc = accuracy_score(y_train, y_pred)
    sen = recall_score(y_train, y_pred)  # Sensitivity is the same as recall
    mcc = matthews_corrcoef(y_train, y_pred)
    f1  = f1_score(y_train, y_pred)
    auc = roc_auc_score(y_train, y_pred)
    bcc = balanced_accuracy_score(y_train, y_pred)
    pre = precision_score(y_train, y_pred)
    # Calculate specificity
    tn, fp, fn, tp = confusion_matrix(y_train, y_pred).ravel()
    spc = tn / (tn + fp)

    # Create a DataFrame to store the metrics
    metrics = pd.DataFrame({
        'BACC': [bcc],
        'Accuracy': [acc],
        'Sensitivity': [sen],
        'Specificity': [spc],
        'MCC': [mcc],
        'AUC': [auc],
        'Precision': [pre],
        'F1 Score': [f1],
    }, index=[col_name])
    return y_pred, metrics

def y_prediction_cv(model, x_train, y_train, col_name):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_preds = []
    fold_metrics = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(x_train, y_train)):
        x_tr, x_val = x_train.iloc[train_idx], x_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        model.fit(x_tr, y_tr)
        y_pred = pd.DataFrame(model.predict(x_val), columns=[col_name], index=x_val.index)
        # Calculate metrics
        sen = recall_score(y_val, y_pred)
        mcc = matthews_corrcoef(y_val, y_pred)
        f1  = f1_score(y_val, y_pred)
        auc = roc_auc_score(y_val, y_pred)
        bcc = balanced_accuracy_score(y_val, y_pred)
        pre = precision_score(y_val, y_pred)
        tn, fp, fn, tp = confusion_matrix(y_val, y_pred).ravel()
        spc = tn / (tn + fp)
        metrics = pd.DataFrame({
            'BACC': [bcc],
            'Sensitivity': [sen],
            'Specificity': [spc],
            'MCC': [mcc],
            'AUC': [auc],
            'Precision': [pre],
            'F1 Score': [f1],
            'Fold': [fold+1]
        }, index=[col_name + f"_fold{fold+1}"])
        # Save predictions and metrics for this fold
        #y_pred.to_csv(f"{col_name}_fold{fold+1}_pred.csv")
        #metrics.to_csv(f"{col_name}_fold{fold+1}_metrics.csv")
        fold_preds.append(y_pred)
        fold_metrics.append(metrics)
    # Aggregate predictions and metrics
    y_pred_all = pd.concat(fold_preds).sort_index()
    metrics_all = pd.concat(fold_metrics)
    # Save all folds together
    y_pred_all.to_csv(f"{col_name}_cv_pred_all.csv")
    metrics_all.to_csv(f"{col_name}_cv_metrics_all.csv")
    return y_pred_all, metrics_all



def y_prediction_loocv(model, x_train, y_train, col_name):
    """
    Perform Leave-One-Out Cross-Validation, collect all predictions, then compute metrics on all predictions at once.
    Metrics are not computed per fold, only for the full set.
    """
    loo = LeaveOneOut()
    preds = []
    indices = []

    for train_idx, val_idx in loo.split(x_train, y_train):
        x_tr, x_val = x_train.iloc[train_idx], x_train.iloc[val_idx]
        y_tr = y_train.iloc[train_idx]
        model.fit(x_tr, y_tr)
        y_val_pred = model.predict(x_val)
        preds.append(y_val_pred[0])
        indices.append(x_val.index[0])

    # Collect all predictions into a DataFrame
    y_pred_all = pd.DataFrame(preds, columns=[col_name], index=indices).sort_index()
    y_true_all = y_train.loc[y_pred_all.index]

    # Compute metrics on all predictions at once
    acc = accuracy_score(y_true_all, y_pred_all)
    sen = recall_score(y_true_all, y_pred_all)
    mcc = matthews_corrcoef(y_true_all, y_pred_all)
    f1  = f1_score(y_true_all, y_pred_all)
    auc_val = roc_auc_score(y_true_all, y_pred_all)
    bcc = balanced_accuracy_score(y_true_all, y_pred_all)
    pre = precision_score(y_true_all, y_pred_all)
    tn, fp, fn, tp = confusion_matrix(y_true_all, y_pred_all).ravel()
    spc = tn / (tn + fp)

    metrics = pd.DataFrame({
        'BACC': [bcc],
        'Accuracy': [acc],
        'Sensitivity': [sen],
        'Specificity': [spc],
        'MCC': [mcc],
        'AUC': [auc_val],
        'Precision': [pre],
        'F1 Score': [f1],
    }, index=[col_name + "_LOO"])

    # Save predictions and metrics
    y_pred_all.to_csv(f"{col_name}_LOO_pred_all.csv")
    metrics.to_csv(f"{col_name}_LOO_metrics_all.csv")

    return y_pred_all, metrics


def plot_auc_auprc_cv(model, x_train, y_train, col_name):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    precisions = []
    auprcs = []
    mean_recall = np.linspace(0, 1, 100)

    # ROC Curve
    plt.figure(figsize=(6, 3))
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
        plt.plot(fpr, tpr, lw=1, alpha=0.7, label=f"Fold {i+1} (AUC = {roc_auc:.2f})")
        # Interpolate tpr
        tprs.append(np.interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
    # Plot mean ROC
    mean_tpr = np.mean(tprs, axis=0)
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    plt.plot(mean_fpr, mean_tpr, color='b', label=f"Mean AUROC = {mean_auc:.2f} ± {std_auc:.2f}", lw=2)
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', lw=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"AUROC - {col_name}")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(f"{col_name}_roc_auc_cv.png"), dpi=500)
    plt.close()

    # Precision-Recall Curve
    plt.figure(figsize=(6, 3))
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
        plt.plot(recall, precision, lw=1, alpha=0.7, label=f"Fold {i+1} (AUPRC = {auprc:.2f})")
        # Interpolate precision
        precisions.append(np.interp(mean_recall, recall[::-1], precision[::-1]))
    # Plot mean PRC
    mean_precision = np.mean(precisions, axis=0)
    mean_auprc = np.mean(auprcs)
    std_auprc = np.std(auprcs)
    plt.plot(mean_recall, mean_precision, color='b', label=f"Mean AUPRC = {mean_auprc:.2f} ± {std_auprc:.2f}", lw=2)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"AUPRC - {col_name}")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(f"{col_name}_prc_auprc_cv.png"), dpi=500)
    plt.close()



def shap_plot(stacked_model, stack_test, name):
    explainer = shap.Explainer(stacked_model)
    shap_values = explainer(stack_test)
    shap.summary_plot(shap_values, stack_test, show=False, plot_type="bar", plot_size=(3, 5))
    plt.xlabel("mean|SHAP|", fontsize=12, fontstyle='italic',weight="bold")
    plt.savefig(name+'_shap.pdf', bbox_inches='tight')
    plt.close()

def stacked_class(name):
    xat_train = pd.read_csv(os.path.join(name,'xat.csv'), index_col=0)
    xes_train = pd.read_csv(os.path.join(name,'xes.csv'), index_col=0)
    xke_train = pd.read_csv(os.path.join(name,'xke.csv'), index_col=0)
    xpc_train = pd.read_csv(os.path.join(name,'xpc.csv'), index_col=0)
    xss_train = pd.read_csv(os.path.join(name,'xss.csv'), index_col=0)
    xcd_train = pd.read_csv(os.path.join(name,'xcd.csv'), index_col=0)
    xcn_train = pd.read_csv(os.path.join(name,'xcn.csv'), index_col=0)
    xkc_train = pd.read_csv(os.path.join(name,'xkc.csv'), index_col=0)
    xce_train = pd.read_csv(os.path.join(name,'xce.csv'), index_col=0)
    xsc_train = pd.read_csv(os.path.join(name,'xsc.csv'), index_col=0)
    xac_train = pd.read_csv(os.path.join(name,'xac.csv'), index_col=0)
    xma_train = pd.read_csv(os.path.join(name,'xma.csv'), index_col=0)
    y_train   = pd.read_csv(os.path.join(name,"y_label.csv"), index_col=0)
    baseline_model_rf_at = RandomForestClassifier(max_depth=5, random_state=1).fit(xat_train, y_train)
    baseline_model_rf_ke = RandomForestClassifier(max_depth=5, random_state=1).fit(xke_train, y_train)
    baseline_model_rf_es = RandomForestClassifier(max_depth=5, random_state=1).fit(xes_train, y_train)
    baseline_model_rf_pc = RandomForestClassifier(max_depth=5, random_state=1).fit(xpc_train, y_train)
    baseline_model_rf_ss = RandomForestClassifier(max_depth=5, random_state=1).fit(xss_train, y_train)
    baseline_model_rf_cd = RandomForestClassifier(max_depth=5, random_state=1).fit(xcd_train, y_train)
    baseline_model_rf_cn = RandomForestClassifier(max_depth=5, random_state=1).fit(xcn_train, y_train)
    baseline_model_rf_kc = RandomForestClassifier(max_depth=5, random_state=1).fit(xkc_train, y_train)
    baseline_model_rf_ce = RandomForestClassifier(max_depth=5, random_state=1).fit(xce_train, y_train)
    baseline_model_rf_sc = RandomForestClassifier(max_depth=5, random_state=1).fit(xsc_train, y_train)
    baseline_model_rf_ac = RandomForestClassifier(max_depth=5, random_state=1).fit(xac_train, y_train)
    baseline_model_rf_ma = RandomForestClassifier(max_depth=5, random_state=1).fit(xma_train, y_train)
    
    dump(baseline_model_rf_at, os.path.join(name, "baseline_model_rf_at.joblib"))
    dump(baseline_model_rf_ke, os.path.join(name, "baseline_model_rf_ke.joblib"))
    dump(baseline_model_rf_es, os.path.join(name, "baseline_model_rf_es.joblib"))
    dump(baseline_model_rf_pc, os.path.join(name, "baseline_model_rf_pc.joblib"))
    dump(baseline_model_rf_ss, os.path.join(name, "baseline_model_rf_ss.joblib"))
    dump(baseline_model_rf_cd, os.path.join(name, "baseline_model_rf_cd.joblib"))
    dump(baseline_model_rf_cn, os.path.join(name, "baseline_model_rf_cn.joblib"))
    dump(baseline_model_rf_kc, os.path.join(name, "baseline_model_rf_kc.joblib"))
    dump(baseline_model_rf_ce, os.path.join(name, "baseline_model_rf_ce.joblib"))
    dump(baseline_model_rf_sc, os.path.join(name, "baseline_model_rf_sc.joblib"))
    dump(baseline_model_rf_ac, os.path.join(name, "baseline_model_rf_ac.joblib"))
    dump(baseline_model_rf_ma, os.path.join(name, "baseline_model_rf_ma.joblib"))
    
    yat_pred_rf_train, yat_metric_rf_train = y_prediction(baseline_model_rf_at, xat_train, y_train, "yat_pred_rf")
    yes_pred_rf_train, yes_metric_rf_train = y_prediction(baseline_model_rf_es, xes_train, y_train, "yes_pred_rf")
    yke_pred_rf_train, yke_metric_rf_train = y_prediction(baseline_model_rf_ke, xke_train, y_train, "yke_pred_rf")
    ypc_pred_rf_train, ypc_metric_rf_train = y_prediction(baseline_model_rf_pc, xpc_train, y_train, "ypc_pred_rf")
    yss_pred_rf_train, yss_metric_rf_train = y_prediction(baseline_model_rf_ss, xss_train, y_train, "yss_pred_rf")
    ycd_pred_rf_train, ycd_metric_rf_train = y_prediction(baseline_model_rf_cd, xcd_train, y_train, "ycd_pred_rf")
    ycn_pred_rf_train, ycn_metric_rf_train = y_prediction(baseline_model_rf_cn, xcn_train, y_train, "ycn_pred_rf")
    ykc_pred_rf_train, ykc_metric_rf_train = y_prediction(baseline_model_rf_kc, xkc_train, y_train, "ykc_pred_rf")
    yce_pred_rf_train, yce_metric_rf_train = y_prediction(baseline_model_rf_ce, xce_train, y_train, "yce_pred_rf")
    ysc_pred_rf_train, ysc_metric_rf_train = y_prediction(baseline_model_rf_sc, xsc_train, y_train, "ysc_pred_rf")
    yac_pred_rf_train, yac_metric_rf_train = y_prediction(baseline_model_rf_ac, xac_train, y_train, "yac_pred_rf")
    yma_pred_rf_train, yma_metric_rf_train = y_prediction(baseline_model_rf_ma, xma_train, y_train, "yma_pred_rf")
    
    yat_pred_rf_cv,    yat_metric_rf_cv  =  y_prediction_cv(baseline_model_rf_at, xat_train, y_train, "yat_pred_rf")
    yes_pred_rf_cv,    yes_metric_rf_cv  =  y_prediction_cv(baseline_model_rf_es, xes_train, y_train, "yes_pred_rf")
    yke_pred_rf_cv,    yke_metric_rf_cv  =  y_prediction_cv(baseline_model_rf_ke, xke_train, y_train, "yke_pred_rf")
    ypc_pred_rf_cv,    ypc_metric_rf_cv  =  y_prediction_cv(baseline_model_rf_pc, xpc_train, y_train, "ypc_pred_rf")
    yss_pred_rf_cv,    yss_metric_rf_cv  =  y_prediction_cv(baseline_model_rf_ss, xss_train, y_train, "yss_pred_rf")
    ycd_pred_rf_cv,    ycd_metric_rf_cv  =  y_prediction_cv(baseline_model_rf_cd, xcd_train, y_train, "ycd_pred_rf")
    ycn_pred_rf_cv,    ycn_metric_rf_cv  =  y_prediction_cv(baseline_model_rf_cn, xcn_train, y_train, "ycn_pred_rf")
    ykc_pred_rf_cv,    ykc_metric_rf_cv  =  y_prediction_cv(baseline_model_rf_kc, xkc_train, y_train, "ykc_pred_rf")
    yce_pred_rf_cv,    yce_metric_rf_cv  =  y_prediction_cv(baseline_model_rf_ce, xce_train, y_train, "yce_pred_rf")
    ysc_pred_rf_cv,    ysc_metric_rf_cv  =  y_prediction_cv(baseline_model_rf_sc, xsc_train, y_train, "ysc_pred_rf")
    yac_pred_rf_cv,    yac_metric_rf_cv  =  y_prediction_cv(baseline_model_rf_ac, xac_train, y_train, "yac_pred_rf")
    yma_pred_rf_cv,    yma_metric_rf_cv  =  y_prediction_cv(baseline_model_rf_ma, xma_train, y_train, "yma_pred_rf")
    
    plot_auc_auprc_cv(baseline_model_rf_at, xat_train, y_train, "yat_pred_rf")
    plot_auc_auprc_cv(baseline_model_rf_es, xes_train, y_train, "yes_pred_rf")
    plot_auc_auprc_cv(baseline_model_rf_ke, xke_train, y_train, "yke_pred_rf")
    plot_auc_auprc_cv(baseline_model_rf_pc, xpc_train, y_train, "ypc_pred_rf")
    plot_auc_auprc_cv(baseline_model_rf_ss, xss_train, y_train, "yss_pred_rf")
    plot_auc_auprc_cv(baseline_model_rf_cd, xcd_train, y_train, "ycd_pred_rf")
    plot_auc_auprc_cv(baseline_model_rf_cn, xcn_train, y_train, "ycn_pred_rf")
    plot_auc_auprc_cv(baseline_model_rf_kc, xkc_train, y_train, "ykc_pred_rf")
    plot_auc_auprc_cv(baseline_model_rf_ce, xce_train, y_train, "yce_pred_rf")
    plot_auc_auprc_cv(baseline_model_rf_sc, xsc_train, y_train, "ysc_pred_rf")
    plot_auc_auprc_cv(baseline_model_rf_ac, xac_train, y_train, "yac_pred_rf")
    plot_auc_auprc_cv(baseline_model_rf_ma, xma_train, y_train, "yma_pred_rf")
    
    yat_pred_rf_loocv,    yat_metric_rf_loocv  =  y_prediction_loocv(baseline_model_rf_at, xat_train, y_train, "yat_pred_rf")
    yes_pred_rf_loocv,    yes_metric_rf_loocv  =  y_prediction_loocv(baseline_model_rf_es, xes_train, y_train, "yes_pred_rf")
    yke_pred_rf_loocv,    yke_metric_rf_loocv  =  y_prediction_loocv(baseline_model_rf_ke, xke_train, y_train, "yke_pred_rf")
    ypc_pred_rf_loocv,    ypc_metric_rf_loocv  =  y_prediction_loocv(baseline_model_rf_pc, xpc_train, y_train, "ypc_pred_rf")
    yss_pred_rf_loocv,    yss_metric_rf_loocv  =  y_prediction_loocv(baseline_model_rf_ss, xss_train, y_train, "yss_pred_rf")
    ycd_pred_rf_loocv,    ycd_metric_rf_loocv  =  y_prediction_loocv(baseline_model_rf_cd, xcd_train, y_train, "ycd_pred_rf")
    ycn_pred_rf_loocv,    ycn_metric_rf_loocv  =  y_prediction_loocv(baseline_model_rf_cn, xcn_train, y_train, "ycn_pred_rf")
    ykc_pred_rf_loocv,    ykc_metric_rf_loocv  =  y_prediction_loocv(baseline_model_rf_kc, xkc_train, y_train, "ykc_pred_rf")
    yce_pred_rf_loocv,    yce_metric_rf_loocv  =  y_prediction_loocv(baseline_model_rf_ce, xce_train, y_train, "yce_pred_rf")
    ysc_pred_rf_loocv,    ysc_metric_rf_loocv  =  y_prediction_loocv(baseline_model_rf_sc, xsc_train, y_train, "ysc_pred_rf")
    yac_pred_rf_loocv,    yac_metric_rf_loocv  =  y_prediction_loocv(baseline_model_rf_ac, xac_train, y_train, "yac_pred_rf")
    yma_pred_rf_loocv,    yma_metric_rf_loocv  =  y_prediction_loocv(baseline_model_rf_ma, xma_train, y_train, "yma_pred_rf")
    #XGB
    baseline_model_xgb_at = xgb.XGBClassifier(objective="binary:logistic",eval_metric='auc', max_depth=5, random_state=1).fit(xat_train, y_train)
    baseline_model_xgb_ke = xgb.XGBClassifier(objective="binary:logistic",eval_metric='auc', max_depth=5, random_state=1).fit(xke_train, y_train)
    baseline_model_xgb_es = xgb.XGBClassifier(objective="binary:logistic",eval_metric='auc', max_depth=5, random_state=1).fit(xes_train, y_train)
    baseline_model_xgb_pc = xgb.XGBClassifier(objective="binary:logistic",eval_metric='auc', max_depth=5, random_state=1).fit(xpc_train, y_train)
    baseline_model_xgb_ss = xgb.XGBClassifier(objective="binary:logistic",eval_metric='auc', max_depth=5, random_state=1).fit(xss_train, y_train)
    baseline_model_xgb_cd = xgb.XGBClassifier(objective="binary:logistic",eval_metric='auc', max_depth=5, random_state=1).fit(xcd_train, y_train)
    baseline_model_xgb_cn = xgb.XGBClassifier(objective="binary:logistic",eval_metric='auc', max_depth=5, random_state=1).fit(xcn_train, y_train)
    baseline_model_xgb_kc = xgb.XGBClassifier(objective="binary:logistic",eval_metric='auc', max_depth=5, random_state=1).fit(xkc_train, y_train)
    baseline_model_xgb_ce = xgb.XGBClassifier(objective="binary:logistic",eval_metric='auc', max_depth=5, random_state=1).fit(xce_train, y_train)
    baseline_model_xgb_sc = xgb.XGBClassifier(objective="binary:logistic",eval_metric='auc', max_depth=5, random_state=1).fit(xsc_train, y_train)
    baseline_model_xgb_ac = xgb.XGBClassifier(objective="binary:logistic",eval_metric='auc', max_depth=5, random_state=1).fit(xac_train, y_train)
    baseline_model_xgb_ma = xgb.XGBClassifier(objective="binary:logistic",eval_metric='auc', max_depth=5, random_state=1).fit(xma_train, y_train)
    dump(baseline_model_xgb_at, os.path.join(name, "baseline_model_xgb_at.joblib"))
    dump(baseline_model_xgb_ke, os.path.join(name, "baseline_model_xgb_ke.joblib"))
    dump(baseline_model_xgb_es, os.path.join(name, "baseline_model_xgb_es.joblib"))
    dump(baseline_model_xgb_pc, os.path.join(name, "baseline_model_xgb_pc.joblib"))
    dump(baseline_model_xgb_ss, os.path.join(name, "baseline_model_xgb_ss.joblib"))
    dump(baseline_model_xgb_cd, os.path.join(name, "baseline_model_xgb_cd.joblib"))
    dump(baseline_model_xgb_cn, os.path.join(name, "baseline_model_xgb_cn.joblib"))
    dump(baseline_model_xgb_kc, os.path.join(name, "baseline_model_xgb_kc.joblib"))
    dump(baseline_model_xgb_ce, os.path.join(name, "baseline_model_xgb_ce.joblib"))
    dump(baseline_model_xgb_sc, os.path.join(name, "baseline_model_xgb_sc.joblib"))
    dump(baseline_model_xgb_ac, os.path.join(name, "baseline_model_xgb_ac.joblib"))
    dump(baseline_model_xgb_ma, os.path.join(name, "baseline_model_xgb_ma.joblib"))
    
    yat_pred_xgb_train, yat_metric_xgb_train = y_prediction(   baseline_model_xgb_at, xat_train, y_train, "yat_pred_xgb")
    yes_pred_xgb_train, yes_metric_xgb_train = y_prediction(   baseline_model_xgb_es, xes_train, y_train, "yes_pred_xgb")
    yke_pred_xgb_train, yke_metric_xgb_train = y_prediction(   baseline_model_xgb_ke, xke_train, y_train, "yke_pred_xgb")
    ypc_pred_xgb_train, ypc_metric_xgb_train = y_prediction(   baseline_model_xgb_pc, xpc_train, y_train, "ypc_pred_xgb")
    yss_pred_xgb_train, yss_metric_xgb_train = y_prediction(   baseline_model_xgb_ss, xss_train, y_train, "yss_pred_xgb")
    ycd_pred_xgb_train, ycd_metric_xgb_train = y_prediction(   baseline_model_xgb_cd, xcd_train, y_train, "ycd_pred_xgb")
    ycn_pred_xgb_train, ycn_metric_xgb_train = y_prediction(   baseline_model_xgb_cn, xcn_train, y_train, "ycn_pred_xgb")
    ykc_pred_xgb_train, ykc_metric_xgb_train = y_prediction(   baseline_model_xgb_kc, xkc_train, y_train, "ykc_pred_xgb")
    yce_pred_xgb_train, yce_metric_xgb_train = y_prediction(   baseline_model_xgb_ce, xce_train, y_train, "yce_pred_xgb")
    ysc_pred_xgb_train, ysc_metric_xgb_train = y_prediction(   baseline_model_xgb_sc, xsc_train, y_train, "ysc_pred_xgb")
    yac_pred_xgb_train, yac_metric_xgb_train = y_prediction(   baseline_model_xgb_ac, xac_train, y_train, "yac_pred_xgb")
    yma_pred_xgb_train, yma_metric_xgb_train = y_prediction(   baseline_model_xgb_ma, xma_train, y_train, "yma_pred_xgb")
    
    plot_auc_auprc_cv(   baseline_model_xgb_at, xat_train, y_train, "yat_pred_xgb")
    plot_auc_auprc_cv(   baseline_model_xgb_es, xes_train, y_train, "yes_pred_xgb")
    plot_auc_auprc_cv(   baseline_model_xgb_ke, xke_train, y_train, "yke_pred_xgb")
    plot_auc_auprc_cv(   baseline_model_xgb_pc, xpc_train, y_train, "ypc_pred_xgb")
    plot_auc_auprc_cv(   baseline_model_xgb_ss, xss_train, y_train, "yss_pred_xgb")
    plot_auc_auprc_cv(   baseline_model_xgb_cd, xcd_train, y_train, "ycd_pred_xgb")
    plot_auc_auprc_cv(   baseline_model_xgb_cn, xcn_train, y_train, "ycn_pred_xgb")
    plot_auc_auprc_cv(   baseline_model_xgb_kc, xkc_train, y_train, "ykc_pred_xgb")
    plot_auc_auprc_cv(   baseline_model_xgb_ce, xce_train, y_train, "yce_pred_xgb")
    plot_auc_auprc_cv(   baseline_model_xgb_sc, xsc_train, y_train, "ysc_pred_xgb")
    plot_auc_auprc_cv(   baseline_model_xgb_ac, xac_train, y_train, "yac_pred_xgb")
    plot_auc_auprc_cv(   baseline_model_xgb_ma, xma_train, y_train, "yma_pred_xgb")
    
    yat_pred_xgb_cv,    yat_metric_xgb_cv    = y_prediction_cv(baseline_model_xgb_at, xat_train, y_train, "yat_pred_xgb")
    yes_pred_xgb_cv,    yes_metric_xgb_cv    = y_prediction_cv(baseline_model_xgb_es, xes_train, y_train, "yes_pred_xgb")
    yke_pred_xgb_cv,    yke_metric_xgb_cv    = y_prediction_cv(baseline_model_xgb_ke, xke_train, y_train, "yke_pred_xgb")
    ypc_pred_xgb_cv,    ypc_metric_xgb_cv    = y_prediction_cv(baseline_model_xgb_pc, xpc_train, y_train, "ypc_pred_xgb")
    yss_pred_xgb_cv,    yss_metric_xgb_cv    = y_prediction_cv(baseline_model_xgb_ss, xss_train, y_train, "yss_pred_xgb")
    ycd_pred_xgb_cv,    ycd_metric_xgb_cv    = y_prediction_cv(baseline_model_xgb_cd, xcd_train, y_train, "ycd_pred_xgb")
    ycn_pred_xgb_cv,    ycn_metric_xgb_cv    = y_prediction_cv(baseline_model_xgb_cn, xcn_train, y_train, "ycn_pred_xgb")
    ykc_pred_xgb_cv,    ykc_metric_xgb_cv    = y_prediction_cv(baseline_model_xgb_kc, xkc_train, y_train, "ykc_pred_xgb")
    yce_pred_xgb_cv,    yce_metric_xgb_cv    = y_prediction_cv(baseline_model_xgb_ce, xce_train, y_train, "yce_pred_xgb")
    ysc_pred_xgb_cv,    ysc_metric_xgb_cv    = y_prediction_cv(baseline_model_xgb_sc, xsc_train, y_train, "ysc_pred_xgb")
    yac_pred_xgb_cv,    yac_metric_xgb_cv    = y_prediction_cv(baseline_model_xgb_ac, xac_train, y_train, "yac_pred_xgb")
    yma_pred_xgb_cv,    yma_metric_xgb_cv    = y_prediction_cv(baseline_model_xgb_ma, xma_train, y_train, "yma_pred_xgb")
    
    yat_pred_xgb_loocv,    yat_metric_xgb_loocv    = y_prediction_loocv(baseline_model_xgb_at, xat_train, y_train, "yat_pred_xgb")
    yes_pred_xgb_loocv,    yes_metric_xgb_loocv    = y_prediction_loocv(baseline_model_xgb_es, xes_train, y_train, "yes_pred_xgb")
    yke_pred_xgb_loocv,    yke_metric_xgb_loocv    = y_prediction_loocv(baseline_model_xgb_ke, xke_train, y_train, "yke_pred_xgb")
    ypc_pred_xgb_loocv,    ypc_metric_xgb_loocv    = y_prediction_loocv(baseline_model_xgb_pc, xpc_train, y_train, "ypc_pred_xgb")
    yss_pred_xgb_loocv,    yss_metric_xgb_loocv    = y_prediction_loocv(baseline_model_xgb_ss, xss_train, y_train, "yss_pred_xgb")
    ycd_pred_xgb_loocv,    ycd_metric_xgb_loocv    = y_prediction_loocv(baseline_model_xgb_cd, xcd_train, y_train, "ycd_pred_xgb")
    ycn_pred_xgb_loocv,    ycn_metric_xgb_loocv    = y_prediction_loocv(baseline_model_xgb_cn, xcn_train, y_train, "ycn_pred_xgb")
    ykc_pred_xgb_loocv,    ykc_metric_xgb_loocv    = y_prediction_loocv(baseline_model_xgb_kc, xkc_train, y_train, "ykc_pred_xgb")
    yce_pred_xgb_loocv,    yce_metric_xgb_loocv    = y_prediction_loocv(baseline_model_xgb_ce, xce_train, y_train, "yce_pred_xgb")
    ysc_pred_xgb_loocv,    ysc_metric_xgb_loocv    = y_prediction_loocv(baseline_model_xgb_sc, xsc_train, y_train, "ysc_pred_xgb")
    yac_pred_xgb_loocv,    yac_metric_xgb_loocv    = y_prediction_loocv(baseline_model_xgb_ac, xac_train, y_train, "yac_pred_xgb")
    yma_pred_xgb_loocv,    yma_metric_xgb_loocv    = y_prediction_loocv(baseline_model_xgb_ma, xma_train, y_train, "yma_pred_xgb")
    #SVC
    baseline_model_svc_at = SVC(random_state=1).fit(xat_train, y_train)
    baseline_model_svc_ke = SVC(random_state=1).fit(xke_train, y_train)
    baseline_model_svc_es = SVC(random_state=1).fit(xes_train, y_train)
    baseline_model_svc_pc = SVC(random_state=1).fit(xpc_train, y_train)
    baseline_model_svc_ss = SVC(random_state=1).fit(xss_train, y_train)
    baseline_model_svc_cd = SVC(random_state=1).fit(xcd_train, y_train)
    baseline_model_svc_cn = SVC(random_state=1).fit(xcn_train, y_train)
    baseline_model_svc_kc = SVC(random_state=1).fit(xkc_train, y_train)
    baseline_model_svc_ce = SVC(random_state=1).fit(xce_train, y_train)
    baseline_model_svc_sc = SVC(random_state=1).fit(xsc_train, y_train)
    baseline_model_svc_ac = SVC(random_state=1).fit(xac_train, y_train)
    baseline_model_svc_ma = SVC(random_state=1).fit(xma_train, y_train)
    dump(baseline_model_svc_at, os.path.join(name, "baseline_model_svc_at.joblib"))
    dump(baseline_model_svc_ke, os.path.join(name, "baseline_model_svc_ke.joblib"))
    dump(baseline_model_svc_es, os.path.join(name, "baseline_model_svc_es.joblib"))
    dump(baseline_model_svc_pc, os.path.join(name, "baseline_model_svc_pc.joblib"))
    dump(baseline_model_svc_ss, os.path.join(name, "baseline_model_svc_ss.joblib"))
    dump(baseline_model_svc_cd, os.path.join(name, "baseline_model_svc_cd.joblib"))
    dump(baseline_model_svc_cn, os.path.join(name, "baseline_model_svc_cn.joblib"))
    dump(baseline_model_svc_kc, os.path.join(name, "baseline_model_svc_kc.joblib"))
    dump(baseline_model_svc_ce, os.path.join(name, "baseline_model_svc_ce.joblib"))
    dump(baseline_model_svc_sc, os.path.join(name, "baseline_model_svc_sc.joblib"))
    dump(baseline_model_svc_ac, os.path.join(name, "baseline_model_svc_ac.joblib"))
    dump(baseline_model_svc_ma, os.path.join(name, "baseline_model_svc_ma.joblib"))
    
    yat_pred_svc_train, yat_metric_svc_train = y_prediction(   baseline_model_svc_at, xat_train, y_train, "yat_pred_svc")
    yes_pred_svc_train, yes_metric_svc_train = y_prediction(   baseline_model_svc_es, xes_train, y_train, "yes_pred_svc")
    yke_pred_svc_train, yke_metric_svc_train = y_prediction(   baseline_model_svc_ke, xke_train, y_train, "yke_pred_svc")
    ypc_pred_svc_train, ypc_metric_svc_train = y_prediction(   baseline_model_svc_pc, xpc_train, y_train, "ypc_pred_svc")
    yss_pred_svc_train, yss_metric_svc_train = y_prediction(   baseline_model_svc_ss, xss_train, y_train, "yss_pred_svc")
    ycd_pred_svc_train, ycd_metric_svc_train = y_prediction(   baseline_model_svc_cd, xcd_train, y_train, "ycd_pred_svc")
    ycn_pred_svc_train, ycn_metric_svc_train = y_prediction(   baseline_model_svc_cn, xcn_train, y_train, "ycn_pred_svc")
    ykc_pred_svc_train, ykc_metric_svc_train = y_prediction(   baseline_model_svc_kc, xkc_train, y_train, "ykc_pred_svc")
    yce_pred_svc_train, yce_metric_svc_train = y_prediction(   baseline_model_svc_ce, xce_train, y_train, "yce_pred_svc")
    ysc_pred_svc_train, ysc_metric_svc_train = y_prediction(   baseline_model_svc_sc, xsc_train, y_train, "ysc_pred_svc")
    yac_pred_svc_train, yac_metric_svc_train = y_prediction(   baseline_model_svc_ac, xac_train, y_train, "yac_pred_svc")
    yma_pred_svc_train, yma_metric_svc_train = y_prediction(   baseline_model_svc_ma, xma_train, y_train, "yma_pred_svc")
    
    plot_auc_auprc_cv(   baseline_model_svc_at, xat_train, y_train, "yat_pred_svc")
    plot_auc_auprc_cv(   baseline_model_svc_es, xes_train, y_train, "yes_pred_svc")
    plot_auc_auprc_cv(   baseline_model_svc_ke, xke_train, y_train, "yke_pred_svc")
    plot_auc_auprc_cv(   baseline_model_svc_pc, xpc_train, y_train, "ypc_pred_svc")
    plot_auc_auprc_cv(   baseline_model_svc_ss, xss_train, y_train, "yss_pred_svc")
    plot_auc_auprc_cv(   baseline_model_svc_cd, xcd_train, y_train, "ycd_pred_svc")
    plot_auc_auprc_cv(   baseline_model_svc_cn, xcn_train, y_train, "ycn_pred_svc")
    plot_auc_auprc_cv(   baseline_model_svc_kc, xkc_train, y_train, "ykc_pred_svc")
    plot_auc_auprc_cv(   baseline_model_svc_ce, xce_train, y_train, "yce_pred_svc")
    plot_auc_auprc_cv(   baseline_model_svc_sc, xsc_train, y_train, "ysc_pred_svc")
    plot_auc_auprc_cv(   baseline_model_svc_ac, xac_train, y_train, "yac_pred_svc")
    plot_auc_auprc_cv(   baseline_model_svc_ma, xma_train, y_train, "yma_pred_svc")
    
    yat_pred_svc_cv,    yat_metric_svc_cv    = y_prediction_cv(baseline_model_svc_at, xat_train, y_train, "yat_pred_svc")
    yes_pred_svc_cv,    yes_metric_svc_cv    = y_prediction_cv(baseline_model_svc_es, xes_train, y_train, "yes_pred_svc")
    yke_pred_svc_cv,    yke_metric_svc_cv    = y_prediction_cv(baseline_model_svc_ke, xke_train, y_train, "yke_pred_svc")
    ypc_pred_svc_cv,    ypc_metric_svc_cv    = y_prediction_cv(baseline_model_svc_pc, xpc_train, y_train, "ypc_pred_svc")
    yss_pred_svc_cv,    yss_metric_svc_cv    = y_prediction_cv(baseline_model_svc_ss, xss_train, y_train, "yss_pred_svc")
    ycd_pred_svc_cv,    ycd_metric_svc_cv    = y_prediction_cv(baseline_model_svc_cd, xcd_train, y_train, "ycd_pred_svc")
    ycn_pred_svc_cv,    ycn_metric_svc_cv    = y_prediction_cv(baseline_model_svc_cn, xcn_train, y_train, "ycn_pred_svc")
    ykc_pred_svc_cv,    ykc_metric_svc_cv    = y_prediction_cv(baseline_model_svc_kc, xkc_train, y_train, "ykc_pred_svc")
    yce_pred_svc_cv,    yce_metric_svc_cv    = y_prediction_cv(baseline_model_svc_ce, xce_train, y_train, "yce_pred_svc")
    ysc_pred_svc_cv,    ysc_metric_svc_cv    = y_prediction_cv(baseline_model_svc_sc, xsc_train, y_train, "ysc_pred_svc")
    yac_pred_svc_cv,    yac_metric_svc_cv    = y_prediction_cv(baseline_model_svc_ac, xac_train, y_train, "yac_pred_svc")
    yma_pred_svc_cv,    yma_metric_svc_cv    = y_prediction_cv(baseline_model_svc_ma, xma_train, y_train, "yma_pred_svc")
    
    yat_pred_svc_loocv,    yat_metric_svc_loocv    = y_prediction_loocv(baseline_model_svc_at, xat_train, y_train, "yat_pred_svc")
    yes_pred_svc_loocv,    yes_metric_svc_loocv    = y_prediction_loocv(baseline_model_svc_es, xes_train, y_train, "yes_pred_svc")
    yke_pred_svc_loocv,    yke_metric_svc_loocv    = y_prediction_loocv(baseline_model_svc_ke, xke_train, y_train, "yke_pred_svc")
    ypc_pred_svc_loocv,    ypc_metric_svc_loocv    = y_prediction_loocv(baseline_model_svc_pc, xpc_train, y_train, "ypc_pred_svc")
    yss_pred_svc_loocv,    yss_metric_svc_loocv    = y_prediction_loocv(baseline_model_svc_ss, xss_train, y_train, "yss_pred_svc")
    ycd_pred_svc_loocv,    ycd_metric_svc_loocv    = y_prediction_loocv(baseline_model_svc_cd, xcd_train, y_train, "ycd_pred_svc")
    ycn_pred_svc_loocv,    ycn_metric_svc_loocv    = y_prediction_loocv(baseline_model_svc_cn, xcn_train, y_train, "ycn_pred_svc")
    ykc_pred_svc_loocv,    ykc_metric_svc_loocv    = y_prediction_loocv(baseline_model_svc_kc, xkc_train, y_train, "ykc_pred_svc")
    yce_pred_svc_loocv,    yce_metric_svc_loocv    = y_prediction_loocv(baseline_model_svc_ce, xce_train, y_train, "yce_pred_svc")
    ysc_pred_svc_loocv,    ysc_metric_svc_loocv    = y_prediction_loocv(baseline_model_svc_sc, xsc_train, y_train, "ysc_pred_svc")
    yac_pred_svc_loocv,    yac_metric_svc_loocv    = y_prediction_loocv(baseline_model_svc_ac, xac_train, y_train, "yac_pred_svc")
    yma_pred_svc_loocv,    yma_metric_svc_loocv    = y_prediction_loocv(baseline_model_svc_ma, xma_train, y_train, "yma_pred_svc")
    
    # Shap EState
    shap_plot(baseline_model_xgb_es, xes_train, "XGB_EState")
    print("finished Baseline with SHAP")
    print("#"*100)
    print("Stacking")
    #stack
    stack_train =pd.concat([yat_pred_rf_train, yat_pred_xgb_train, yat_pred_svc_train,
                            yes_pred_rf_train, yes_pred_xgb_train, yes_pred_svc_train,
                            yke_pred_rf_train, yke_pred_xgb_train, yke_pred_svc_train,
                            ypc_pred_rf_train, ypc_pred_xgb_train, ypc_pred_svc_train,
                            yss_pred_rf_train, yss_pred_xgb_train, yss_pred_svc_train,
                            ycd_pred_rf_train, ycd_pred_xgb_train, ycd_pred_svc_train,
                            ycn_pred_rf_train, ycn_pred_xgb_train, ycn_pred_svc_train,
                            ykc_pred_rf_train, ykc_pred_xgb_train, ykc_pred_svc_train,
                            yce_pred_rf_train, yce_pred_xgb_train, yce_pred_svc_train,
                            ysc_pred_rf_train, ysc_pred_xgb_train, ysc_pred_svc_train,
                            yac_pred_rf_train, yac_pred_xgb_train, yac_pred_svc_train,
                            yma_pred_rf_train, yma_pred_xgb_train, yma_pred_svc_train,],  axis=1)
    stack_cv    =pd.concat([yat_pred_rf_cv,    yat_pred_xgb_cv,    yat_pred_svc_cv,
                            yes_pred_rf_cv,    yes_pred_xgb_cv,    yes_pred_svc_cv,
                            yke_pred_rf_cv,    yke_pred_xgb_cv,    yke_pred_svc_cv,
                            ypc_pred_rf_cv,    ypc_pred_xgb_cv,    ypc_pred_svc_cv,
                            yss_pred_rf_cv,    yss_pred_xgb_cv,    yss_pred_svc_cv,
                            ycd_pred_rf_cv,    ycd_pred_xgb_cv,    ycd_pred_svc_cv,
                            ycn_pred_rf_cv,    ycn_pred_xgb_cv,    ycn_pred_svc_cv,
                            ykc_pred_rf_cv,    ykc_pred_xgb_cv,    ykc_pred_svc_cv,
                            yce_pred_rf_cv,    yce_pred_xgb_cv,    yce_pred_svc_cv,
                            ysc_pred_rf_cv,    ysc_pred_xgb_cv,    ysc_pred_svc_cv,
                            yac_pred_rf_cv,    yac_pred_xgb_cv,    yac_pred_svc_cv,
                            yma_pred_rf_cv,    yma_pred_xgb_cv,    yma_pred_svc_cv,],  axis=1)
    stack_loocv =pd.concat([yat_pred_rf_loocv,  yat_pred_xgb_loocv,  yat_pred_svc_loocv,
                            yes_pred_rf_loocv,  yes_pred_xgb_loocv,  yes_pred_svc_loocv,
                            yke_pred_rf_loocv,  yke_pred_xgb_loocv,  yke_pred_svc_loocv,
                            ypc_pred_rf_loocv,  ypc_pred_xgb_loocv,  ypc_pred_svc_loocv,
                            yss_pred_rf_loocv,  yss_pred_xgb_loocv,  yss_pred_svc_loocv,
                            ycd_pred_rf_loocv,  ycd_pred_xgb_loocv,  ycd_pred_svc_loocv,
                            ycn_pred_rf_loocv,  ycn_pred_xgb_loocv,  ycn_pred_svc_loocv,
                            ykc_pred_rf_loocv,  ykc_pred_xgb_loocv,  ykc_pred_svc_loocv,
                            yce_pred_rf_loocv,  yce_pred_xgb_loocv,  yce_pred_svc_loocv,
                            ysc_pred_rf_loocv,  ysc_pred_xgb_loocv,  ysc_pred_svc_loocv,
                            yac_pred_rf_loocv,  yac_pred_xgb_loocv,  yac_pred_svc_loocv,
                            yma_pred_rf_loocv,  yma_pred_xgb_loocv,  yma_pred_svc_loocv,],  axis=1)

    
    #save stack original
    stack_train.to_csv(os.path.join( name, 'stack', "stacked_train.csv"))
    stack_cv.to_csv(os.path.join(    name, 'stack', "stacked_cv.csv"))
    stack_loocv.to_csv(os.path.join(  name, 'stack', "stacked_test.csv"))
    
    #Train
    stacked_xgb = xgb.XGBClassifier(objective="binary:logistic",eval_metric='auc', random_state=1)
    
    param_grid = {
    'max_depth': [5]
    }
    
    # Set up GridSearchCV
    grid_search = GridSearchCV(estimator=stacked_xgb, param_grid=param_grid, scoring='roc_auc', cv=5, verbose=1)
    grid_search.fit(stack_train, y_train)
    best_params = grid_search.best_params_
    
    print("Best parameters found: ", best_params)
    print("Best AUC found: ", grid_search.best_score_)
    
    stacked_model = xgb.XGBClassifier(**best_params)
    stacked_model.set_params(objective='binary:logistic', eval_metric='auc', random_state=1)
    stacked_model.fit(stack_train, y_train)
    
    dump(stacked_model, os.path.join(name, 'stack', "stacked_model.joblib"))

    y_pred_stk_train,  y_metric_stk_train = y_prediction(   stacked_model, stack_train, y_train, "y_pred_stacked")
    y_pred_stk_cv   ,  y_metric_stk_cv    = y_prediction(   stacked_model, stack_cv,    y_train, "y_pred_stacked")
    y_pred_stk_loocv ,  y_metric_stk_loocv  = y_prediction(   stacked_model, stack_loocv, y_train,  "y_pred_stacked")
    
    y_pred_stk_train.to_csv(os.path.join( name, 'stack', "y_pred_train.csv"))
    y_pred_stk_loocv.to_csv(os.path.join( name, 'stack', "y_pred_test.csv"))
    y_pred_stk_cv   .to_csv(os.path.join( name, 'stack', "y_pred_cv.csv"))
    
    #combine metrics
    metric_train= pd.concat([yat_metric_rf_train, 
                            yac_metric_rf_train,   
                            ycn_metric_rf_train,   
                            yce_metric_rf_train,   
                            ycd_metric_rf_train,   
                            yes_metric_rf_train,   
                            yke_metric_rf_train,   
                            ykc_metric_rf_train,   
                            yma_metric_rf_train,   
                            ypc_metric_rf_train,   
                            yss_metric_rf_train,   
                            ysc_metric_rf_train,   
                            yat_metric_svc_train,
                            yac_metric_svc_train,
                            ycn_metric_svc_train,
                            yce_metric_svc_train,
                            ycd_metric_svc_train,
                            yes_metric_svc_train,
                            yke_metric_svc_train,
                            ykc_metric_svc_train,
                            yma_metric_svc_train,
                            ypc_metric_svc_train,
                            yss_metric_svc_train,
                            ysc_metric_svc_train,
                            yat_metric_xgb_train,
                            yac_metric_xgb_train,
                            ycn_metric_xgb_train,
                            yce_metric_xgb_train,
                            ycd_metric_xgb_train,
                            yes_metric_xgb_train,
                            yke_metric_xgb_train,
                            ykc_metric_xgb_train,
                            yma_metric_xgb_train,
                            ypc_metric_xgb_train,
                            yss_metric_xgb_train,
                            ysc_metric_xgb_train,
                            y_metric_stk_train],  axis=0)
    
    metric_cv  = pd.concat([yat_metric_rf_cv, 
                            yac_metric_rf_cv, 
                            ycn_metric_rf_cv, 
                            yce_metric_rf_cv, 
                            ycd_metric_rf_cv, 
                            yes_metric_rf_cv, 
                            yke_metric_rf_cv, 
                            ykc_metric_rf_cv, 
                            yma_metric_rf_cv, 
                            ypc_metric_rf_cv, 
                            yss_metric_rf_cv, 
                            ysc_metric_rf_cv,  
                            yat_metric_svc_cv,
                            yac_metric_svc_cv,
                            ycn_metric_svc_cv,
                            yce_metric_svc_cv,
                            ycd_metric_svc_cv,
                            yes_metric_svc_cv,
                            yke_metric_svc_cv,
                            ykc_metric_svc_cv,
                            yma_metric_svc_cv,
                            ypc_metric_svc_cv,
                            yss_metric_svc_cv,
                            ysc_metric_svc_cv,
                            yat_metric_xgb_cv,  
                            yac_metric_xgb_cv,  
                            ycn_metric_xgb_cv,  
                            yce_metric_xgb_cv,  
                            ycd_metric_xgb_cv,  
                            yes_metric_xgb_cv,  
                            yke_metric_xgb_cv,  
                            ykc_metric_xgb_cv,  
                            yma_metric_xgb_cv,  
                            ypc_metric_xgb_cv,  
                            yss_metric_xgb_cv,  
                            ysc_metric_xgb_cv,  
                            y_metric_stk_cv],  axis=0)
    
    metric_loocv  = pd.concat([yat_metric_rf_loocv, 
                            yac_metric_rf_loocv, 
                            ycn_metric_rf_loocv, 
                            yce_metric_rf_loocv, 
                            ycd_metric_rf_loocv, 
                            yes_metric_rf_loocv, 
                            yke_metric_rf_loocv, 
                            ykc_metric_rf_loocv, 
                            yma_metric_rf_loocv, 
                            ypc_metric_rf_loocv, 
                            yss_metric_rf_loocv, 
                            ysc_metric_rf_loocv,  
                            yat_metric_svc_loocv,
                            yac_metric_svc_loocv,
                            ycn_metric_svc_loocv,
                            yce_metric_svc_loocv,
                            ycd_metric_svc_loocv,
                            yes_metric_svc_loocv,
                            yke_metric_svc_loocv,
                            ykc_metric_svc_loocv,
                            yma_metric_svc_loocv,
                            ypc_metric_svc_loocv,
                            yss_metric_svc_loocv,
                            ysc_metric_svc_loocv,
                            yat_metric_xgb_loocv,  
                            yac_metric_xgb_loocv,  
                            ycn_metric_xgb_loocv,  
                            yce_metric_xgb_loocv,  
                            ycd_metric_xgb_loocv,  
                            yes_metric_xgb_loocv,  
                            yke_metric_xgb_loocv,  
                            ykc_metric_xgb_loocv,  
                            yma_metric_xgb_loocv,  
                            ypc_metric_xgb_loocv,  
                            yss_metric_xgb_loocv,  
                            ysc_metric_xgb_loocv,  
                            y_metric_stk_loocv],  axis=0)
    
    # round number
    metric_train = round(metric_train, 3)
    metric_cv    = round(metric_cv, 3)
    metric_loocv = round(metric_loocv, 3)
    
    metric_train.to_csv(os.path.join( name, "metric_train.csv"))
    metric_cv   .to_csv(os.path.join( name, "metric_cv.csv"))
    metric_loocv.to_csv(os.path.join( name, "metric_loocv.csv"))
    return stacked_model, stack_train, stack_cv, stack_loocv, metric_train, metric_loocv, metric_cv, best_params


def nearest_neighbor_AD(x_train, x_test, name, k, z=3):
    from sklearn.neighbors import NearestNeighbors
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
    # Calculate di of test
    distance, index = nn.kneighbors(x_test)
    di = np.mean(distance, axis=1)
    AD_status = ['within_AD' if di[i] < dk + (z * sk) else 'outside_AD' for i in range(len(di))]

    # Create DataFrame with index from x_test and the respective status
    df = pd.DataFrame(AD_status, index=x_test.index, columns=['AD_status'])
    return df, dk, sk

def run_ad(stacked_model, stack_train, stack_test, y_train, name, z = 0.5):
    # Initialize lists to store metrics for plotting
    k_values = [3, 4, 5, 6, 7, 8, 9, 10]
    MCC_values = []
    ACC_values = []
    Sen_values = []
    Spe_values = []
    AUC_values = []
    F1_values = []
    BA_values = []
    Pre_values = []
    removed_compounds_values = []
    dk_values = []
    sk_values = []
    
    # Remove outside AD
    for i in k_values:
        print('k = ', i, 'z=', str(z))
        t, dk, sk = nearest_neighbor_AD(stack_train, stack_test, name, i, z=z)
        t.to_csv("AD_train_set_"+str(i)+".csv")
        print(t['AD_status'].value_counts())
        # Remove outside AD
        x_ad_test = stack_test[t['AD_status'] == 'within_AD']
        y_ad_test = y_test.loc[x_ad_test.index]
        y_pred_test = stacked_model.predict(x_ad_test)
        print(len(x_ad_test),len(y_ad_test), len(y_pred_test) )
        # Evaluation
        print('Test set')
        accuracy = round(accuracy_score(y_ad_test, y_pred_test), 3)
        conf_matrix = confusion_matrix(y_ad_test, y_pred_test)
        F1 = round(f1_score(y_ad_test, y_pred_test, average='weighted'), 3)
        tn, fp, fn, tp = conf_matrix.ravel()
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)
        auc = roc_auc_score(y_ad_test, y_pred_test)
        mcc = round(matthews_corrcoef(y_ad_test, y_pred_test), 3)
        balanced_acc = round(balanced_accuracy_score(y_ad_test, y_pred_test), 3)
        pre_scores = round(precision_score(y_ad_test, y_pred_test), 3)
        print('ACC: ', accuracy, 'Sen: ', sensitivity, 'Spe: ', specificity, 'MCC: ', mcc,'AUC: ', auc,'BA: ', balanced_acc, 'Pre:', pre_scores, 'F1: ', F1)
        # Store metrics for plotting
        MCC_values.append(mcc)
        ACC_values.append(accuracy)
        Sen_values.append(sensitivity)
        Spe_values.append(specificity)
        AUC_values.append(auc)
        F1_values.append(F1)
        BA_values.append(balanced_acc)
        Pre_values.append(pre_scores)
        removed_compounds_values.append((t['AD_status'] == 'outside_AD').sum())
        dk_values.append(dk)
        sk_values.append(sk)
    k_values   = np.array(k_values)
    MCC_values = np.array(MCC_values)
    ACC_values = np.array(ACC_values)
    Sen_values = np.array(Sen_values)
    Spe_values = np.array(Spe_values)
    AUC_values = np.array(AUC_values)
    F1_values  = np.array(F1_values)
    BA_values  = np.array(BA_values)
    Pre_values = np.array(Pre_values)
    dk_values  = np.array(dk_values)
    sk_values  = np.array(sk_values)
    removed_compounds_values = np.array(removed_compounds_values)
    # Save table
    ad_metrics = pd.DataFrame({
        "k": k_values[:len(MCC_values)],  # Adjust if some values are skipped
        "Accuracy": ACC_values,
        "Balanced Accuracy": BA_values,
        "Sensitivity": Sen_values,
        "Specificity": Spe_values,
        "MCC": MCC_values,
        "AUC": AUC_values,
        "Precision": Pre_values,
        "F1 Score": F1_values,
        "Removed Compounds": removed_compounds_values,
        "dk_values": dk_values,
        "sk_values": sk_values
    })
    ad_metrics = round(ad_metrics, 3)
    ad_metrics.to_csv("AD_metrics_"+name+"_"+ str(z)+ ".csv")
    # Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 3))
    
    ax1.plot(k_values, BA_values,  'bo-',  label = "BACC")
    ax1.plot(k_values, Sen_values, 'gs-', label = "Sensitivity")
    ax1.plot(k_values, Spe_values, 'y*-', label = "Specificity")
    ax1.plot(k_values, MCC_values, 'r^-', label = "MCC")
    ax1.plot(k_values, AUC_values, 'md-', label = "AUC")
    ax1.plot(k_values, AUC_values, 'cD-', label = "Precision")
    ax1.plot(k_values, F1_values,  'cX-',  label = "F1")
    # Adding labels and title
    ax1.set_xlabel('k',      fontsize=12, fontstyle='italic',weight="bold")
    ax1.set_ylabel('Scores', fontsize=12, fontstyle='italic', weight='bold')
    ax1.set_xticks(k_values)
    ax1.legend(loc='upper left', fontsize='small', bbox_to_anchor=(1.05, 1.02))
    # Second plot: Bar plot for removed_compounds_values
    ax2.bar(k_values, removed_compounds_values, color='green', edgecolor='black', alpha=0.5, width=0.3)
    ax2.set_xlabel('k', fontsize=12, fontstyle='italic',weight="bold")
    ax2.set_ylabel('Removed compounds', fontsize=12, fontstyle='italic', weight='bold')
    ax2.set_xticks(k_values)
    plt.tight_layout()
    plt.savefig("AD_"+name+"_"+ str(z)+ "_Classification_separated.svg", bbox_inches='tight') 
    plt.close

def y_random(stack_train, stack_cv, stack_test, y_train, y_test, metric_train, metric_test, best_params, name):
    MCC_test=[]
    MCC_train=[]
    for i in range(1,101):
      y_train=y_train.sample(frac=1,replace=False,random_state=0)

      model=xgb.XGBClassifier(**best_params).fit(stack_cv, y_train)
      y_pred_MCCext=model.predict(stack_test)
      y_pred_MCCtrain=model.predict(stack_train)
      MCCext=matthews_corrcoef(y_test, y_pred_MCCext)
      MCC_test.append(MCCext)
      MCCtrain=matthews_corrcoef(y_train, y_pred_MCCtrain)
      MCC_train.append(MCCtrain)
    size=[50]
    sizes=[20]
    x=[metric_train.loc['y_pred_stacked', 'MCC']]
    y=[metric_test.loc['y_pred_stacked', 'MCC']]
    fig, ax = plt.subplots(figsize=(3, 3))
    ax.axvline(0.5, c='black', ls=':')
    ax.axhline(0.5, c='black', ls=':')
    ax.scatter(x,y,s=size,c=['red'],marker='x', label='Our model')
    ax.scatter(MCC_train,MCC_test, c='blue',edgecolors='black', alpha=0.7, s=sizes, label='Y-randomization')
    ax.set_xlabel('$MCC_{Train}$', fontsize=14,  fontstyle='italic', weight='bold')
    ax.set_ylabel('$MCC_{Test}$', fontsize=14,  fontstyle='italic', weight='bold')
    ax.legend(loc='lower right',fontsize='small')
    # Adjust layout
    plt.tight_layout()
    plt.savefig("Y-randomization-"+name+"-classification.pdf", bbox_inches='tight')
    # Show the plots
    plt.close()

def plot_importance_xgb(model, name):
    importance = model.get_booster().get_score(importance_type='gain')
    features_df = pd.DataFrame(list(importance.items()), columns=['Feature', 'Importance']).sort_values(by='Importance', ascending=False)
    print(features_df.head(10))
    plt.figure(figsize=(4, 4))
    sns.barplot(x='Importance', y='Feature', data=features_df)
    plt.xlabel("Importance scores", fontsize=12, fontstyle='italic',weight="bold")
    plt.ylabel("Features", fontsize=12, fontstyle='italic',weight="bold")
    plt.tight_layout()
    plt.savefig('top_features_xgb_'+name+'.pdf', bbox_inches='tight')
    plt.close()
    
def shap_plot(stacked_model, stack_test, name):
    explainer = shap.Explainer(stacked_model)
    shap_values = explainer(stack_test)
    shap.summary_plot(shap_values, stack_test, show=False, plot_type="bar", plot_size=(3, 5))
    plt.xlabel("mean|SHAP|", fontsize=12, fontstyle='italic',weight="bold")
    plt.savefig(name+'_shap.pdf', bbox_inches='tight')
    plt.close()
    
def main():
    for name in ['nafld']:  
        print("#"*100) 
        print(name)
        y_train  = pd.read_csv(os.path.join(name,"y_label.csv"), index_col=0)
        print("Y_train")
        print(y_train)
        stacked_model, stack_train, stack_cv, stack_loocv, metric_train, metric_loocv, metric_cv, best_params = stacked_class(name)
        print("finish train ", name)
        #shap_plot(stacked_model, stack_train, "XGB_stacked")
        #print("finished shap", name)
        #y_random(stack_train, stack_cv, stack_test, y_train, y_test, metric_train, metric_test, best_params, name)
        #print("finish yrandom ", name)
        #plot_importance_xgb(stacked_model, name)
        #print("finish top features ", name)
        #run_ad(stacked_model, stack_cv, stack_test, y_test, name, z=0.5)
        #print("finish ad ", name)
        
if __name__ == "__main__":
    main()

