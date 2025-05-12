import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, f1_score, recall_score, confusion_matrix, roc_curve, auc, precision_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.utils.class_weight import compute_class_weight

# Function to fit logistic regression
def fit_logistic_regression(df, independent_vars, dep_var, sample_weights=None):
    X = df[independent_vars]
    X = sm.add_constant(X)
    model = sm.Logit(dep_var, X).fit(disp=0, weights=sample_weights)

    summary = model.summary()

    ORs = np.exp(model.params)
    OR_CI_lower = np.exp(model.conf_int()[0])
    OR_CI_upper = np.exp(model.conf_int()[1])
    p_values = model.pvalues

    OR_results = pd.DataFrame({
        'Odds Ratio': ORs,
        '95% CI Lower': OR_CI_lower,
        '95% CI Upper': OR_CI_upper,
        'p-value': p_values
    })

    return model, summary, OR_results

# Function to compute metrics
def compute_metrics(y_true, y_pred_probs, threshold=0.5):

    y_pred = (y_pred_probs >= threshold).astype(int)
    auc_value = roc_auc_score(y_true, y_pred_probs)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    sensitivity = recall_score(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0

    return auc_value, bal_acc, sensitivity, specificity, precision, f1

# Function to find best threshold
def find_best_threshold(y_true, y_pred_probs):
    thresholds = np.arange(0.1, 0.9, 0.01)
    best_threshold = 0.5
    best_balanced_acc = -1

    for threshold in thresholds:
        y_pred = (y_pred_probs >= threshold).astype(int)
        bal_acc = balanced_accuracy_score(y_true, y_pred)

        if bal_acc > best_balanced_acc:
            best_balanced_acc = bal_acc
            best_threshold = threshold

    return best_threshold

# Function for bootstrapping
def bootstrap_metrics(df_train, df_test, independent_vars, dep_var_train, dep_var_test, n_bootstrap=1000, threshold=None, sample_weights=None):
    # Automatically compute sample weights if not provided
    if sample_weights is None:
        classes = np.unique(dep_var_train)
        weights = compute_class_weight(class_weight='balanced', classes=classes, y=dep_var_train)
        sample_weights = dep_var_train.map(dict(zip(classes, weights)))
        print("Using class weighting automatically.")

    model, summary, OR_results = fit_logistic_regression(df_train, independent_vars, dep_var_train, sample_weights=sample_weights)

    y_pred_probs_full_test = model.predict(sm.add_constant(df_test[independent_vars], has_constant='add'))

    if threshold is None:
        threshold = find_best_threshold(dep_var_test, y_pred_probs_full_test)
        print(f"Best Threshold found automatically: {threshold:.2f}")

    boot_auc, boot_bal_acc, boot_sens, boot_spec, boot_prec, boot_f1 = [], [], [], [], [], []

    sss = StratifiedShuffleSplit(n_splits=n_bootstrap, test_size=len(df_test)-5, random_state=42)

    for _, test_index in sss.split(df_test, dep_var_test):
        sample_df_test = df_test.iloc[test_index]
        sample_dep_test = dep_var_test.iloc[test_index]

        y_pred_probs = model.predict(sm.add_constant(sample_df_test[independent_vars], has_constant='add'))

        auc_value, bal_acc, sens, spec, prec, f1 = compute_metrics(sample_dep_test, y_pred_probs, threshold)

        boot_auc.append(auc_value)
        boot_bal_acc.append(bal_acc)
        boot_sens.append(sens)
        boot_spec.append(spec)
        boot_prec.append(prec)
        boot_f1.append(f1)

    def ci(data):
        return np.mean(data), np.percentile(data, [2.5, 97.5])

    return {
        "model": model,
        "Odds Ratios (one-time fit)": OR_results,
        "Regression Summary": summary,
        "Best Threshold": threshold,
        "Bootstrapped Metrics": {
            "AUC": ci(boot_auc),
            "Balanced Accuracy": ci(boot_bal_acc),
            "Sensitivity": ci(boot_sens),
            "Specificity": ci(boot_spec),
            "Precision": ci(boot_prec),
            "F1-score": ci(boot_f1),
        },
        "Arrays with bootstrapping": {
            "AUC": boot_auc,
            "Balanced Accuracy": boot_bal_acc,
            "Sensitivity": boot_sens,
            "Specificity": boot_spec,
            "Precision": boot_prec,
            "F1-score": boot_f1,
        },
        "True Labels": dep_var_test,
        "Predicted Probabilities": y_pred_probs_full_test
    }

# Function to compute adjusted OR
def compute_adjusted_or(model_result, increase=0.1):
    coef = model_result.params
    conf_int = model_result.conf_int()

    odds_ratios = np.exp(coef)
    adjusted_or = np.exp(coef * increase)
    conf_int_lower = np.exp(conf_int[0] * increase)
    conf_int_upper = np.exp(conf_int[1] * increase)

    or_df = pd.DataFrame({
        "Odds Ratio (1 unit)": odds_ratios,
        f"Adjusted OR ({increase} increase)": adjusted_or,
        "95% CI Lower": conf_int_lower,
        "95% CI Upper": conf_int_upper,
        "p-value": model_result.pvalues
    })

    return or_df

# Function to plot ROC Curve
def plot_roc_curve(y_true, y_pred_probs, title='ROC Curve'):
    fpr, tpr, _ = roc_curve(y_true, y_pred_probs)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, lw=2, label=f'AUC = {roc_auc:.4f}')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=1)

    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])

    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.title(f"{title}\n(AUC = {roc_auc:.4f})", fontsize=16)
    plt.legend(loc='lower right', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

# Function to build summary table
def build_summary_table(results_list):
    summary_rows = []

    for result in results_list:
        row = {
            "Category": result['Category'],
            "Best Threshold": result.get('Best Threshold', None)
        }

        for metric_name, (mean, ci) in result["Bootstrapped Metrics"].items():
            row[f"{metric_name} Mean"] = round(mean, 4)
            row[f"{metric_name} 95% CI Lower"] = round(ci[0], 4)
            row[f"{metric_name} 95% CI Upper"] = round(ci[1], 4)

        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    return summary_df