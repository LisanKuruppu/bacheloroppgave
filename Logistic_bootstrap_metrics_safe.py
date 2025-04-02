
import numpy as np
import pandas as pd
import statsmodels.api as sm

def fit_logistic_regression(df, independent_vars, dep_var):
    X = df[independent_vars]
    
    def safe_fit_logit(X, y):
        tried_cols = X.columns.tolist()
        while True:
            try:
                X_const = sm.add_constant(X[tried_cols])
                model = sm.Logit(y, X_const).fit(disp=0)
                return model, tried_cols
            except np.linalg.LinAlgError:
                dropped = tried_cols.pop()
                print(f"⚠️ Singular matrix error. Dropping '{dropped}' and retrying...")
                if len(tried_cols) < 2:
                    raise Exception("Not enough features left to fit a model.")
    
    model, used_cols = safe_fit_logit(X, dep_var)
    X = X[used_cols]  # restrict to usable features
    summary = model.summary()
    
    conf = model.conf_int()
    conf.columns = ['2.5%', '97.5%']
    ORs = pd.DataFrame({
        'OR': np.exp(model.params),
        '2.5%': np.exp(conf['2.5%']),
        '97.5%': np.exp(conf['97.5%']),
        'p-value': model.pvalues
    })
    return model, summary, ORs

def compute_metrics(y_true, y_pred_probs, threshold=0.5):
    y_pred = (y_pred_probs >= threshold).astype(int)
    from sklearn.metrics import roc_auc_score, balanced_accuracy_score, recall_score, f1_score, confusion_matrix

    auc = roc_auc_score(y_true, y_pred_probs)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    sensitivity = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1]) if (cm[0, 0] + cm[0, 1]) != 0 else 0

    return auc, bal_acc, sensitivity, specificity, f1

def bootstrap_metrics(df_train, df_test, independent_vars, dep_var_train, dep_var_test, n_bootstrap=1000, threshold=0.5):
    model, summary, OR_results = fit_logistic_regression(df_train, independent_vars, dep_var_train)

    boot_auc, boot_bal_acc, boot_sens, boot_spec, boot_f1 = [], [], [], [], []

    for _ in range(n_bootstrap):
        resample = df_train.sample(frac=1.0, replace=True)
        y_resample = dep_var_train.loc[resample.index]
        
        try:
            model_i, _, _ = fit_logistic_regression(resample, independent_vars, y_resample)
            y_pred_probs = model_i.predict(sm.add_constant(df_test[independent_vars]))
            metrics = compute_metrics(dep_var_test, y_pred_probs, threshold)
            boot_auc.append(metrics[0])
            boot_bal_acc.append(metrics[1])
            boot_sens.append(metrics[2])
            boot_spec.append(metrics[3])
            boot_f1.append(metrics[4])
        except:
            continue

    def mean_ci(lst):
        return np.mean(lst), (np.percentile(lst, 2.5), np.percentile(lst, 97.5))

    return {
        "Odds Ratios (one-time fit)": OR_results,
        "Bootstrapped Metrics": {
            "ROC-AUC": mean_ci(boot_auc),
            "Balanced Accuracy": mean_ci(boot_bal_acc),
            "Sensitivity": mean_ci(boot_sens),
            "Specificity": mean_ci(boot_spec),
            "F1-score": mean_ci(boot_f1),
        }
    }
