import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, f1_score, recall_score, confusion_matrix
from sklearn.model_selection import StratifiedShuffleSplit

# Function to fit logistic regression
def fit_logistic_regression(df, independent_vars, dep_var):
    """Fits logistic regression and returns predicted probabilities"""
    X = df[independent_vars]
    X = sm.add_constant(X)  # Add intercept
    
    model = sm.Logit(dep_var, X).fit(disp=0)  # Suppress output
    
    summary = model.summary()  # Full statsmodels summary

    # Extract ORs and their 95% CI
    ORs = np.exp(model.params)
    OR_CI_lower = np.exp(model.conf_int()[0])  # Lower bound of 95% CI
    OR_CI_upper = np.exp(model.conf_int()[1])  # Upper bound of 95% CI
    p_values = model.pvalues  # P-values for each coefficient
    
    OR_results = pd.DataFrame({
        'Odds Ratio': ORs,
        '95% CI Lower': OR_CI_lower,
        '95% CI Upper': OR_CI_upper,
        'p-value': p_values
    })
    
    return model, summary, OR_results

# Function to compute metrics
def compute_metrics(y_true, y_pred_probs, threshold=0.5):
    """Computes AUC, Balanced Accuracy, Sensitivity, Specificity, and F1-score"""
    y_pred = (y_pred_probs >= threshold).astype(int)  # Convert probabilities to binary

    auc = roc_auc_score(y_true, y_pred_probs)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    sensitivity = recall_score(y_true, y_pred)  # Sensitivity (Recall)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0  # Avoid division by zero

    return auc, bal_acc, sensitivity, specificity, f1

# Function for bootstrapping
def bootstrap_metrics(df_train, df_test, independent_vars, dep_var_train, dep_var_test, n_bootstrap=1000, threshold=0.5):
    """Performs bootstrap resampling and collects all metrics"""
    
    # Fit model once and get ORs
    model, summary, OR_results = fit_logistic_regression(df_train, independent_vars, dep_var_train)
    
    boot_auc, boot_bal_acc, boot_sens, boot_spec, boot_f1 = [], [], [], [], []
    
    # Stratified resampling (to ensure label distribution is maintained in each bootstrap sample)
    sss = StratifiedShuffleSplit(n_splits=n_bootstrap, test_size=len(df_test)-5, random_state=42)
    
    for _, test_index in sss.split(df_test, dep_var_test):  # Splitting indices based on stratification
        # Get the stratified bootstrap sample
        sample_df_test = df_test.iloc[test_index]
        sample_dep_test = dep_var_test.iloc[test_index]
        
        # Get predicted probabilities for the resampled test data
        y_pred_probs = model.predict(sm.add_constant(sample_df_test[independent_vars]))
        
        # Compute performance metrics
        auc, bal_acc, sens, spec, f1 = compute_metrics(sample_dep_test, y_pred_probs, threshold)
        
        # Append metrics to the lists
        boot_auc.append(auc)
        boot_bal_acc.append(bal_acc)
        boot_sens.append(sens)
        boot_spec.append(spec)
        boot_f1.append(f1)
    
    # Compute means and 95% Confidence Intervals
    def ci(data):
        return np.mean(data), np.percentile(data, [2.5, 97.5])

    return {
        "model": model,
        "Odds Ratios (one-time fit)": OR_results,  # Extract OR table
        "Regression Summary": summary,  # Full regression summary
        "Bootstrapped Metrics": {
            "AUC": ci(boot_auc),
            "Balanced Accuracy": ci(boot_bal_acc),
            "Sensitivity": ci(boot_sens),
            "Specificity": ci(boot_spec),
            "F1-score": ci(boot_f1),
        },
        "Arrays with bootstrapping":{
            "AUC": boot_auc,
            "Balanced Accuracy": boot_bal_acc,
            "Sensitivity": boot_sens,
            "Specificity": boot_spec,
            "F1-score": boot_f1,
        }
    }

def compute_adjusted_or(model_result, increase=0.1):
    """
    Compute adjusted odds ratio (OR) for a given increase in the predictor variable.
    
    Parameters:
    - model_result: Fitted statsmodels Logit model
    - increase: The increment for which we compute the OR (default = 0.1)
    
    Returns:
    - DataFrame with OR, adjusted OR for the given increase, and confidence intervals.
    """
    # Extract coefficients and confidence intervals
    coef = model_result.params  # Log-odds coefficients
    conf_int = model_result.conf_int()  # 95% CI in log-odds scale
    
    # Compute standard OR (for a full 1-unit increase)
    odds_ratios = np.exp(coef)
    
    # Compute adjusted OR for the specified increase
    adjusted_or = np.exp(coef * increase)  # OR for smaller increase (e.g., 0.1)
    
    # Compute 95% CI for the adjusted OR
    conf_int_lower = np.exp(conf_int[0] * increase)
    conf_int_upper = np.exp(conf_int[1] * increase)
    
    # Combine into a dataframe for easy viewing
    or_df = pd.DataFrame({
        "Odds Ratio (1 unit)": odds_ratios,
        f"Adjusted OR ({increase} increase)": adjusted_or,
        "95% CI Lower": conf_int_lower,
        "95% CI Upper": conf_int_upper,
        "p-value": model_result.pvalues
    })
    
    return or_df



#### Example usage:
"""
### Your train set
X_train = train_data_model[['TAU']]  # Features for Model 2
y_train = train_data_model['Clinical_AD'].loc[train_data_model.index]  # Target variable for Model 2 (e.g., clinical AD vs Non-AD)

# Check if data contains NaN or infinite values
if not check_data_validity(X_train, y_train):
    raise ValueError("Invalid data detected. Please check the training data.")
    
# Check if X_train and y_train have data
print("\n===== Checking train shape =====\n")
print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")
print("\n================================\n")

# Add constant to the features for intercept in logistic regression model
X_train_with_intercept = sm.add_constant(X_train)

# Step 1: Fit logistic regression model for Model 1
try:
    logit_model = sm.Logit(y_train, X_train_with_intercept)
    logit_result = logit_model.fit(disp=0)
except Exception as e:
    print(f"Error fitting logistic regression model for Model 1: {e}")
    raise

# Step 3: Print the summaries of both models
print("Model 1 Summary:")
print(logit_result.summary())

### Your test set
X_test = df_DL_test[['TAU']].dropna()  # Model 2 data
y_test = df_DL_test['Clinical_AD'].loc[X_test.index]

print("\n===== Checking test shape =====\n")
# Check if X_train and y_train have data
print(f"X_test shape: {X_test.shape}")
print(f"y_test shape: {y_test.shape}")
print("\n================================\n")

X_test_with_intercept = sm.add_constant(X_test)

# Predict probabilities on the test set for Model 1
y_pred_prob_test = logit_result.predict(X_test_with_intercept)

# Step 5: Calculate ROC-AUC for both models
roc_auc_test = roc_auc_score(y_test, y_pred_prob_test)

print(f"ROC-AUC for univariate DL model: {roc_auc_test:.4f}")

# Step 6: Find the best threshold for Model 1 based on balanced accuracy
thresholds = np.arange(0.1, 1.0, 0.1)
best_threshold_model = None
best_balanced_accuracy_model = -np.inf

for threshold in thresholds:
    y_pred_class = (y_pred_prob_test >= threshold).astype(int)
    conf_matrix = confusion_matrix(y_test, y_pred_class)
    sensitivity = conf_matrix[1, 1] / (conf_matrix[1, 0] + conf_matrix[1, 1])
    specificity = conf_matrix[0, 0] / (conf_matrix[0, 0] + conf_matrix[0, 1])
    balanced_accuracy = (sensitivity + specificity) / 2
    
    if balanced_accuracy > best_balanced_accuracy_model:
        best_balanced_accuracy_model = balanced_accuracy
        best_threshold_model = threshold
        
print("\n===== Optimizing threshold =====\n")
print(f"Best Threshold: {best_threshold_model:.1f}, Best Balanced Accuracy: {best_balanced_accuracy_model:.4f}")
print("\n================================\n")

# Step 8: Apply the bootstrap 
bootstrap_results_model = bootstrap_metrics(X_train, X_test, ['TAU'], y_train, y_test, n_bootstrap=1000, threshold=best_threshold_model)

print("\n===== Summary bootstrapping n = 1000 =====\n")
print(bootstrap_results_model["Odds Ratios (one-time fit)"])
print("\n===== Summary bootstrapping metrics n = 1000 =====\n")
print(bootstrap_results_model["Bootstrapped Metrics"])
print("\n================================\n")

# Compute OR for 0.1 increase
or_results = compute_adjusted_or(bootstrap_results_model["model"], increase=0.1)
print("\n===== Adjusted Odds Ratios =====\n")
print(or_results)
print("\n================================\n")

"""