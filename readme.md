# Stacked-NAFLD: An Accurate Stacking Model Targeting In Vivo NAFLD Treatment
# Step 1: Preprocessing
- Canonicalize SMILES:
  → `from revised folder/preprocess.py import canonical_smiles`
  → `df = canonical_smiles(df, smiles_column='SMILES')`

- Compute 13 fingerprints:
  → `from revised folder /preprocess.py import compute_fps`
  → `fps_df = compute_fps(df)`
  → Save to `revised folder/nafld/(fingerprint name.csv`)

# Step 2: Feature Cleaning
- Remove constant string descriptors:
  → `from revised folder/train.py import remove_constant_string_des` (to clean constant value across all fp)
  → `df = remove_constant_string_des(df)`

- Remove highly correlated features:
  → `from revised folder/train.py import remove_highly_correlated_features` 
  → `df = remove_highly_correlated_features(df, threshold=0.7)`
  → Save to `revised folder/nafld/(fingerprint name_reduced.csv`)

# Step 3: Stacked Model Training and Evaluation

- Define base models and molecular fingerprints  
  (e.g., Random Forest, XGBoost, SVM using ECFP, MACCS, PubChem, etc.)

- Generate out-of-fold (OOF) predictions for each base model  
  → These are used as input features for the meta-model (stacked_features)

- Train meta-model using XGBoost  
  → `from revised/train.py import y_prediction`  
  → `y_prediction(model, x_train, y_train, 'Model_Name')`

- Evaluate using cross-validation  
  → `from revised/train.py import y_prediction_cv`  
  → `y_prediction_cv(model, x_train, y_train, 'Model_Name')`

- Evaluate using leave-one-out cross-validation (LOOCV)  
  → `from revised/train.py import y_prediction_loocv`  
  → `y_prediction_loocv(model, x_train, y_train, 'Model_Name')`

- Save outputs for downstream analysis:  
  → `results_cv.csv`  
  → `results_loocv.csv`  
  → `stacked_features.csv`  

# Step 4: Evaluation & Interpretation
- AUROC and AUPRC plot:
  → `from folder/train.py import plot_auc_auprc_cv`
  → `plot_auc_auprc_cv(model, x_train, y_train, 'Model_Name')`
   → Save to `revised folder/graph_metrics

- SHAP plot:
  → `from revised .folder/train.py import shap_plot`
  → `shap_plot(stacked_model, stack_test, 'Model_Name')`
  → Save to `revised folder/New_results_MetaNAFLD_Revised1

# Step 5: Applicability Domain (AD)
- Nearest neighbor AD:
  → `from revised folder/train.py import nearest_neighbor_AD`
  → `nearest_neighbor_AD(x_train, 'Model_Name', k=5, z=3)`

- AD with CV:
  → `from revised folder/train.py import run_ad_cv`
  → `run_ad_cv(stacked_model, stack_train, y_train, 'Model_Name', z=3)`
  → Save to `revised folder/AD_CV_metrics.scv


- AD with full set:
  → `from revised folder/train.py import run_ad`
  → `run_ad(stacked_model, stack_train, y_train, 'Model_Name', z=0.5)`

# Step 6: Y-Randomization
- With AUROC/AUPRC:
  → `from revised folder/train.py import y_random_auroc_auprc`
  → `y_random_auroc_auprc(stacked_features, y, metric_cv, metric_loocv, best_params, 'Model_Name')`
  → Save to `revised folder/Y-randomization-Stacked_XGB-AUROC-AUPRC.pdf
# Step 7 : SHAP analysis
 → `from revised folder/highlight.py
 → `highlight_substructures(smiles, substructure_dict, compound_name="compound")
 → Save to `revised folder/molecules_highlight
This repository store code for reproduce this work.

MIT License
Copyright (c) [2025] [Dr.Tarapong Srisongram]

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
