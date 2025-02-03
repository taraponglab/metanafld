def shap_plot(stacked_model, stack_test, name):
    explainer = shap.TreeExplainer(stacked_model)
    shap_values = explainer.shap_values(stack_test)
    shap.summary_plot(shap_values, stack_test, show=False, plot_size=(3, 5))
    plt.xlabel("mean|SHAP|", fontsize=12, fontstyle='italic',weight="bold")
    plt.savefig('shap_classification_'+name+'.svg', bbox_inches='tight')
    plt.close()

def plot_top_features(model, feature_names, name, title='Top 20 Feature Importances'):
    coefficients = model.coef_[0]  # Assuming it's a simple logistic regression model
    feature_importances = list(zip(feature_names, coefficients))
    feature_importances = sorted(feature_importances, key=lambda x: np.abs(x[1]), reverse=True)
    top_features = feature_importances[:20]
    top_feature_names, top_coefficients = zip(*top_features)
    fig, ax = plt.subplots(figsize=(3, 5))
    y_pos = np.arange(len(top_feature_names))
    ax.barh(y_pos, top_coefficients, align='center', color='blue')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_feature_names)
    ax.invert_yaxis()  # labels read top-to-bottom
    ax.set_xlabel('Coefficient Magnitude')
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig('top_features'+name+'.svg', bbox_inches='tight')
    plt.close()