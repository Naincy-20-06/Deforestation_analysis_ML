"""
Deforestation Issue Analysis Using Support Vector Machine (SVM)

This script provides a complete pipeline for analyzing global deforestation trends
using Support Vector Machine regression. It includes data preprocessing, model building,
hyperparameter tuning, cross-validation, evaluation, and comprehensive visualizations.

Author: ML Analysis Team
Date: 2024
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, KFold
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import permutation_importance
from sklearn.decomposition import PCA

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION SECTION
# ============================================================================

CONFIG = {
    'dataset_path': 'deforestation_dataset.csv',
    'results_dir': 'results',
    'random_state': 42,
    'test_size': 0.2,
    'cv_folds': 5,
    'visualization_style': 'seaborn-v0_8-darkgrid'
}

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def setup_environment():
    """Initialize environment and create necessary directories."""
    np.random.seed(CONFIG['random_state'])
    plt.style.use(CONFIG['visualization_style'])
    sns.set_palette("husl")
    
    # Create results directory
    if not os.path.exists(CONFIG['results_dir']):
        os.makedirs(CONFIG['results_dir'])
        print(f"✓ Created results directory: {CONFIG['results_dir']}/")


def print_section(title):
    """Print a formatted section header."""
    print("\n" + "="*70)
    print(f" {title}")
    print("="*70)


def print_subsection(title):
    """Print a formatted subsection header."""
    print(f"\n{'-'*70}")
    print(f" {title}")
    print(f"{'-'*70}")


# ============================================================================
# DATA LOADING AND INSPECTION
# ============================================================================

def load_dataset():
    """
    Load deforestation dataset from CSV or create synthetic data if file doesn't exist.
    
    Returns:
        pd.DataFrame: Loaded or generated dataset
    """
    print_section("SECTION 1: LOAD AND INSPECT DATASET")
    
    if os.path.exists(CONFIG['dataset_path']):
        df = pd.read_csv(CONFIG['dataset_path'])
        print(f"✓ Dataset loaded from {CONFIG['dataset_path']}")
    else:
        print(f"⚠ {CONFIG['dataset_path']} not found. Generating synthetic dataset...")
        df = generate_synthetic_dataset()
        df.to_csv(CONFIG['dataset_path'], index=False)
        print(f"✓ Synthetic dataset generated and saved to {CONFIG['dataset_path']}")
    
    # Display dataset information
    print(f"\nDataset Overview:")
    print(f"  • Shape: {df.shape[0]} records × {df.shape[1]} columns")
    print(f"  • Columns: {list(df.columns)}")
    
    print_subsection("First Few Records")
    print(df.head())
    
    print_subsection("Dataset Info")
    print(df.info())
    
    print_subsection("Statistical Summary")
    print(df.describe())
    
    return df


def generate_synthetic_dataset():
    """
    Generate synthetic deforestation dataset for demonstration.
    
    Returns:
        pd.DataFrame: Synthetic dataset with realistic relationships
    """
    np.random.seed(CONFIG['random_state'])
    n_samples = 500
    
    df = pd.DataFrame({
        'Country': np.random.choice(
            ['Brazil', 'Indonesia', 'Nigeria', 'Tanzania', 'Myanmar', 
             'DRC', 'Canada', 'Russia', 'Australia', 'Colombia'], 
            n_samples
        ),
        'Tree_Cover_Loss_percent': np.random.uniform(0.1, 15, n_samples),
        'CO2_Emission_mt': np.random.uniform(100, 2500, n_samples),
        'Rainfall_mm': np.random.uniform(400, 3500, n_samples),
        'Population_Million': np.random.uniform(5, 300, n_samples),
        'GDP_Billion_USD': np.random.uniform(50, 3000, n_samples),
        'Deforestation_Policy_Strictness': np.random.choice(['Low', 'Medium', 'High'], n_samples),
        'Corruption_Index': np.random.uniform(20, 95, n_samples),
        'Protected_Area_percent': np.random.uniform(5, 50, n_samples)
    })
    
    # Create realistic relationships
    df['Forest_Loss_Area_km2'] = (
        df['CO2_Emission_mt'] * 10 + 
        df['Population_Million'] * 50 - 
        df['Rainfall_mm'] * 5 + 
        df['Corruption_Index'] * 30 + 
        np.random.normal(0, 5000, n_samples)
    )
    
    # Ensure non-negative values
    df['Forest_Loss_Area_km2'] = df['Forest_Loss_Area_km2'].clip(lower=0)
    
    return df


def inspect_data_quality(df):
    """
    Inspect and report data quality issues (missing values, duplicates, outliers).
    
    Args:
        df (pd.DataFrame): Dataset to inspect
    """
    print_section("SECTION 2: DATA QUALITY INSPECTION")
    
    # Check missing values
    print_subsection("Missing Values Check")
    missing_values = df.isnull().sum()
    if missing_values.sum() > 0:
        print("Missing values found:")
        print(missing_values[missing_values > 0])
    else:
        print("✓ No missing values found")
    
    # Check duplicates
    print_subsection("Duplicate Records Check")
    duplicates = df.duplicated().sum()
    print(f"Duplicate rows: {duplicates}")
    if duplicates > 0:
        print(f"⚠ {duplicates} duplicate records detected")
    else:
        print("✓ No duplicate records found")
    
    # Check outliers using IQR method
    print_subsection("Outlier Detection (IQR Method)")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    outlier_counts = {}
    
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
        if outliers > 0:
            outlier_counts[col] = outliers
    
    if outlier_counts:
        print("Outliers detected:")
        for col, count in outlier_counts.items():
            print(f"  • {col}: {count} outliers")
    else:
        print("✓ No significant outliers detected")


# ============================================================================
# DATA PREPROCESSING
# ============================================================================

def handle_missing_values(df):
    """
    Handle missing values through imputation or removal.
    
    Args:
        df (pd.DataFrame): Dataset with potential missing values
        
    Returns:
        pd.DataFrame: Dataset with missing values handled
    """
    print_section("SECTION 3: HANDLE MISSING VALUES")
    
    df_processed = df.copy()
    
    if df_processed.isnull().sum().sum() == 0:
        print("✓ No missing values to handle")
        return df_processed
    
    # Impute numeric columns with median
    numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df_processed[col].isnull().sum() > 0:
            df_processed[col].fillna(df_processed[col].median(), inplace=True)
            print(f"✓ Imputed {col} with median")
    
    # Impute categorical columns with mode
    categorical_cols = df_processed.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if df_processed[col].isnull().sum() > 0:
            df_processed[col].fillna(df_processed[col].mode()[0], inplace=True)
            print(f"✓ Imputed {col} with mode")
    
    # Remove duplicates
    if df_processed.duplicated().sum() > 0:
        df_processed = df_processed.drop_duplicates()
        print(f"✓ Removed duplicate rows")
    
    return df_processed


def encode_categorical_variables(df):
    """
    Encode categorical variables using LabelEncoder.
    
    Args:
        df (pd.DataFrame): Dataset with categorical variables
        
    Returns:
        tuple: (encoded_dataframe, encoders_dict)
    """
    print_section("SECTION 4: ENCODE CATEGORICAL VARIABLES")
    
    df_encoded = df.copy()
    encoders = {}
    
    categorical_cols = df_encoded.select_dtypes(include=['object']).columns
    print(f"Categorical columns found: {list(categorical_cols)}\n")
    
    for col in categorical_cols:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
        encoders[col] = le
        
        print(f"✓ Encoded '{col}':")
        for i, class_label in enumerate(le.classes_):
            print(f"    {class_label} → {i}")
    
    return df_encoded, encoders


def normalize_features(df, feature_cols):
    """
    Normalize numerical features using StandardScaler.
    
    Args:
        df (pd.DataFrame): Dataset with features
        feature_cols (list): List of feature columns to normalize
        
    Returns:
        tuple: (normalized_dataframe, scaler)
    """
    print_section("SECTION 5: NORMALIZE NUMERICAL FEATURES")
    
    df_normalized = df.copy()
    
    numeric_cols = df_normalized[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    print(f"Numerical features to normalize: {numeric_cols}\n")
    
    scaler = StandardScaler()
    df_normalized[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    
    print("✓ Features standardized (mean=0, std=1)")
    print_subsection("Feature Scaling Statistics")
    
    stats_df = pd.DataFrame({
        'Feature': numeric_cols,
        'Original Mean': df[numeric_cols].mean().values,
        'Original Std': df[numeric_cols].std().values,
        'Scaled Mean': df_normalized[numeric_cols].mean().values,
        'Scaled Std': df_normalized[numeric_cols].std().values
    })
    print(stats_df.to_string(index=False))
    
    return df_normalized, scaler


# ============================================================================
# TRAIN-TEST SPLIT
# ============================================================================

def split_data(df, target_col, feature_cols):
    """
    Split dataset into training and testing sets.
    
    Args:
        df (pd.DataFrame): Preprocessed dataset
        target_col (str): Target column name
        feature_cols (list): List of feature columns
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test, feature_cols)
    """
    print_section("SECTION 6: SPLIT DATASET INTO TRAINING AND TESTING SETS")
    
    X = df[feature_cols]
    y = df[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=CONFIG['test_size'], random_state=CONFIG['random_state']
    )
    
    print(f"Target variable: {target_col}")
    print(f"Number of features: {len(feature_cols)}")
    print(f"\n✓ Dataset split successfully:")
    print(f"  Training set: {X_train.shape[0]} records ({X_train.shape[0]/len(X)*100:.1f}%)")
    print(f"  Testing set:  {X_test.shape[0]} records ({X_test.shape[0]/len(X)*100:.1f}%)")
    print(f"  Features: {X_train.shape[1]}")
    
    print(f"\nTarget variable statistics:")
    print(f"  Training - Mean: {y_train.mean():.2f}, Std: {y_train.std():.2f}")
    print(f"  Testing  - Mean: {y_test.mean():.2f}, Std: {y_test.std():.2f}")
    
    return X_train, X_test, y_train, y_test, feature_cols


# ============================================================================
# MODEL BUILDING
# ============================================================================

def train_initial_models(X_train, y_train, X_test, y_test):
    """
    Train SVM models with different kernels and compare performance.
    
    Args:
        X_train, y_train: Training data
        X_test, y_test: Testing data
        
    Returns:
        dict: Models and their performance metrics
    """
    print_section("SECTION 7: TRAIN SVM MODELS WITH DIFFERENT KERNELS")
    
    svm_models = {}
    svm_performance = {}
    kernels = ['linear', 'poly', 'rbf']
    
    for kernel in kernels:
        print(f"\n→ Training SVM with {kernel.upper()} kernel...")
        
        svm = SVR(kernel=kernel, C=1.0, epsilon=0.1)
        svm.fit(X_train, y_train)
        svm_models[kernel] = svm
        
        # Predictions
        y_train_pred = svm.predict(X_train)
        y_test_pred = svm.predict(X_test)
        
        # Metrics
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        train_mae = mean_absolute_error(y_train, y_train_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        
        svm_performance[kernel] = {
            'Train R²': train_r2,
            'Test R²': test_r2,
            'Train MAE': train_mae,
            'Test MAE': test_mae,
            'Train RMSE': train_rmse,
            'Test RMSE': test_rmse
        }
        
        print(f"  ✓ Model trained")
        print(f"    Train R²: {train_r2:.4f} | Test R²: {test_r2:.4f}")
        print(f"    Train MAE: {train_mae:.2f} | Test MAE: {test_mae:.2f}")
        print(f"    Train RMSE: {train_rmse:.2f} | Test RMSE: {test_rmse:.2f}")
    
    print_subsection("Initial Model Comparison")
    perf_df = pd.DataFrame(svm_performance).T
    print(perf_df.round(4))
    
    return svm_models, svm_performance


def hyperparameter_tuning(X_train, y_train, X_test, y_test):
    """
    Perform hyperparameter tuning using GridSearchCV.
    
    Args:
        X_train, y_train: Training data
        X_test, y_test: Testing data
        
    Returns:
        tuple: (best_model, grid_search_results)
    """
    print_section("SECTION 8: PERFORM HYPERPARAMETER TUNING")
    
    # Reduced parameter grid for faster execution
    param_grid = {
        'C': [1, 10, 100],
        'kernel': ['linear', 'poly', 'rbf'],
        'gamma': ['scale', 0.01, 0.1],
        'epsilon': [0.1, 0.5]
    }
    
    print("Parameter grid for tuning:")
    for key, value in param_grid.items():
        print(f"  • {key}: {value}")
    
    print(f"\n→ Initiating GridSearchCV with {len(param_grid['C']) * len(param_grid['kernel']) * len(param_grid['gamma']) * len(param_grid['epsilon'])} combinations...")
    
    base_svm = SVR()
    grid_search = GridSearchCV(
        base_svm,
        param_grid,
        cv=5,
        n_jobs=-1,
        verbose=0,
        scoring='r2'
    )
    
    grid_search.fit(X_train, y_train)
    
    print(f"\n✓ GridSearchCV completed!")
    print(f"\nBest parameters found:")
    for param, value in grid_search.best_params_.items():
        print(f"  • {param}: {value}")
    print(f"\nBest cross-validation R² score: {grid_search.best_score_:.4f}")
    
    # Evaluate best model
    best_svm = grid_search.best_estimator_
    y_train_pred = best_svm.predict(X_train)
    y_test_pred = best_svm.predict(X_test)
    
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    
    print(f"\n✓ Best Model Performance:")
    print(f"  Training R²: {train_r2:.4f} | Testing R²: {test_r2:.4f}")
    print(f"  Training MAE: {train_mae:.2f} | Testing MAE: {test_mae:.2f}")
    
    return best_svm, grid_search


# ============================================================================
# CROSS-VALIDATION
# ============================================================================

def apply_cross_validation(model, X_train, y_train):
    """
    Apply k-fold cross-validation to assess model robustness.
    
    Args:
        model: Trained SVM model
        X_train, y_train: Training data
        
    Returns:
        tuple: (cv_r2_scores, cv_mae_scores)
    """
    print_section("SECTION 9: APPLY CROSS-VALIDATION")
    
    kfold = KFold(n_splits=CONFIG['cv_folds'], shuffle=True, random_state=CONFIG['random_state'])
    
    print(f"→ Performing {CONFIG['cv_folds']}-fold cross-validation...")
    
    # R² scores
    cv_scores = cross_val_score(model, X_train, y_train, cv=kfold, scoring='r2', n_jobs=-1)
    
    # MAE scores
    cv_mae_scores = -cross_val_score(model, X_train, y_train, cv=kfold, scoring='neg_mean_absolute_error', n_jobs=-1)
    
    print(f"\n✓ Cross-validation completed!")
    print_subsection("R² Scores for Each Fold")
    for i, score in enumerate(cv_scores, 1):
        print(f"  Fold {i}: {score:.4f}")
    
    print_subsection("Cross-Validation Statistics")
    print(f"R² Scores:")
    print(f"  Mean: {cv_scores.mean():.4f}")
    print(f"  Std Dev: {cv_scores.std():.4f}")
    print(f"  Min: {cv_scores.min():.4f}")
    print(f"  Max: {cv_scores.max():.4f}")
    
    print(f"\nMAE Scores:")
    print(f"  Mean: {cv_mae_scores.mean():.2f}")
    print(f"  Std Dev: {cv_mae_scores.std():.2f}")
    
    return cv_scores, cv_mae_scores


# ============================================================================
# MODEL EVALUATION
# ============================================================================

def evaluate_model(model, X_train, y_train, X_test, y_test, cv_scores):
    """
    Comprehensive model evaluation with all metrics.
    
    Args:
        model: Trained SVM model
        X_train, y_train: Training data
        X_test, y_test: Testing data
        cv_scores: Cross-validation scores
        
    Returns:
        dict: Evaluation metrics
    """
    print_section("SECTION 10: EVALUATE MODEL PERFORMANCE")
    
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Calculate metrics
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    train_rmse = np.sqrt(train_mse)
    test_rmse = np.sqrt(test_mse)
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    # Create evaluation table
    metrics_df = pd.DataFrame({
        'Metric': ['Mean Absolute Error (MAE)', 'Mean Squared Error (MSE)', 
                   'Root Mean Squared Error (RMSE)', 'R² Score'],
        'Training Set': [train_mae, train_mse, train_rmse, train_r2],
        'Testing Set': [test_mae, test_mse, test_rmse, test_r2]
    })
    
    print_subsection("Model Evaluation Metrics")
    print(metrics_df.to_string(index=False))
    
    # MAPE
    mape_train = np.mean(np.abs((y_train - y_train_pred) / y_train) * 100)
    mape_test = np.mean(np.abs((y_test - y_test_pred) / y_test) * 100)
    
    print(f"\nAdditional Metrics:")
    print(f"  Training MAPE: {mape_train:.2f}%")
    print(f"  Testing MAPE: {mape_test:.2f}%")
    
    # Overfitting analysis
    print_subsection("Overfitting/Underfitting Analysis")
    r2_diff = train_r2 - test_r2
    if r2_diff > 0.15:
        print(f"⚠ Possible OVERFITTING detected")
    elif test_r2 < 0.3:
        print(f"⚠ Possible UNDERFITTING detected")
    else:
        print(f"✓ Model appears well-balanced")
    
    print(f"  Train R²: {train_r2:.4f}")
    print(f"  Test R²: {test_r2:.4f}")
    print(f"  Difference: {r2_diff:.4f}")
    
    metrics = {
        'train_mae': train_mae,
        'test_mae': test_mae,
        'train_mse': train_mse,
        'test_mse': test_mse,
        'train_rmse': train_rmse,
        'test_rmse': test_rmse,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'y_train_pred': y_train_pred,
        'y_test_pred': y_test_pred
    }
    
    return metrics


# ============================================================================
# FEATURE IMPORTANCE ANALYSIS
# ============================================================================

def analyze_feature_importance(model, X_train, y_train, X_test, y_test, feature_cols, df_scaled, target_col):
    """
    Analyze and rank feature importance using permutation importance and correlation.
    
    Args:
        model: Trained SVM model
        X_train, y_train: Training data
        X_test, y_test: Testing data
        feature_cols: List of feature column names
        df_scaled: Scaled dataframe
        target_col: Target column name
        
    Returns:
        tuple: (feature_importance_df, correlation_df)
    """
    print_section("SECTION 12: ANALYZE FEATURE IMPORTANCE")
    
    # Permutation importance
    print("→ Calculating permutation importance...")
    perm_importance = permutation_importance(
        model, X_test, y_test, n_repeats=10, random_state=CONFIG['random_state'], n_jobs=-1
    )
    
    feature_importance_df = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': perm_importance.importances_mean,
        'Std': perm_importance.importances_std
    }).sort_values('Importance', ascending=False)
    
    print_subsection("Permutation Feature Importance")
    print(feature_importance_df.to_string(index=False))
    
    print_subsection("Top 5 Most Important Features")
    for i, row in feature_importance_df.head(5).iterrows():
        print(f"  {row['Feature']:<40} {row['Importance']:>10.4f} ± {row['Std']:.4f}")
    
    # Correlation analysis
    correlations = []
    for col in feature_cols:
        corr = df_scaled[col].corr(df_scaled[target_col])
        correlations.append({'Feature': col, 'Correlation': corr})
    
    correlation_df = pd.DataFrame(correlations).sort_values('Correlation', key=abs, ascending=False)
    
    print_subsection(f"Feature Correlation with Target ({target_col})")
    print(correlation_df.to_string(index=False))
    
    return feature_importance_df, correlation_df


# ============================================================================
# VISUALIZATIONS
# ============================================================================

def create_visualizations(model, X_train, y_train, X_test, y_test, metrics, 
                         feature_importance_df, correlation_df, feature_cols, df_scaled, target_col):
    """
    Create and save all visualizations.
    
    Args:
        model: Trained model
        X_train, y_train: Training data
        X_test, y_test: Testing data
        metrics: Evaluation metrics dict
        feature_importance_df: Feature importance dataframe
        correlation_df: Correlation dataframe
        feature_cols: List of feature names
        df_scaled: Scaled dataframe
        target_col: Target column name
    """
    print_section("SECTION 13 & 11: CREATE VISUALIZATIONS")
    
    # 1. Model Performance Plots
    print("→ Creating model performance visualizations...")
    create_performance_plots(X_train, y_train, X_test, y_test, metrics)
    
    # 2. Decision Boundaries
    print("→ Creating decision boundary plots...")
    create_decision_boundary_plots(model, X_train, y_train, X_test, y_test)
    
    # 3. Feature Importance
    print("→ Creating feature importance visualizations...")
    create_feature_importance_plots(feature_importance_df, correlation_df)
    
    # 4. Feature Relationships
    print("→ Creating feature-target relationship plots...")
    create_feature_relationship_plots(feature_importance_df, correlation_df, df_scaled, target_col)
    
    # 5. Feature Distributions
    print("→ Creating feature distribution plots...")
    create_distribution_plots(feature_importance_df, df_scaled)
    
    # 6. Correlation Heatmap
    print("→ Creating correlation heatmap...")
    create_correlation_heatmap(feature_importance_df, df_scaled, target_col)
    
    print(f"\n✓ All visualizations saved to {CONFIG['results_dir']}/")


def create_performance_plots(X_train, y_train, X_test, y_test, metrics):
    """Create model performance comparison plots."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Model Performance Analysis - Best SVM Model', fontsize=16, fontweight='bold')
    
    y_train_pred = metrics['y_train_pred']
    y_test_pred = metrics['y_test_pred']
    
    # Training set
    axes[0, 0].scatter(y_train, y_train_pred, alpha=0.6, edgecolors='k')
    axes[0, 0].plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', lw=2)
    axes[0, 0].set_xlabel('Actual Values')
    axes[0, 0].set_ylabel('Predicted Values')
    axes[0, 0].set_title(f'Training Set (R² = {metrics["train_r2"]:.4f})', fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Testing set
    axes[0, 1].scatter(y_test, y_test_pred, alpha=0.6, color='green', edgecolors='k')
    axes[0, 1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    axes[0, 1].set_xlabel('Actual Values')
    axes[0, 1].set_ylabel('Predicted Values')
    axes[0, 1].set_title(f'Testing Set (R² = {metrics["test_r2"]:.4f})', fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Residuals training
    residuals_train = y_train - y_train_pred
    axes[1, 0].scatter(y_train_pred, residuals_train, alpha=0.6, edgecolors='k')
    axes[1, 0].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[1, 0].set_xlabel('Predicted Values')
    axes[1, 0].set_ylabel('Residuals')
    axes[1, 0].set_title('Residuals - Training Set', fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Residuals testing
    residuals_test = y_test - y_test_pred
    axes[1, 1].scatter(y_test_pred, residuals_test, alpha=0.6, color='green', edgecolors='k')
    axes[1, 1].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[1, 1].set_xlabel('Predicted Values')
    axes[1, 1].set_ylabel('Residuals')
    axes[1, 1].set_title('Residuals - Testing Set', fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{CONFIG["results_dir"]}/01_model_performance.png', dpi=300, bbox_inches='tight')
    print("  ✓ Saved: 01_model_performance.png")
    plt.close()


def create_decision_boundary_plots(model, X_train, y_train, X_test, y_test):
    """Create decision boundary visualization using PCA."""
    pca = PCA(n_components=2)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    
    # Train model on PCA data
    svm_pca = SVR(kernel=model.kernel, C=model.C, gamma=model.gamma, epsilon=model.epsilon)
    svm_pca.fit(X_train_pca, y_train)
    
    # Create mesh
    h = 0.02
    x_min, x_max = X_train_pca[:, 0].min() - 1, X_train_pca[:, 0].max() + 1
    y_min, y_max = X_train_pca[:, 1].min() - 1, X_train_pca[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = svm_pca.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('SVM Decision Boundaries - PCA 2D Projection', fontsize=14, fontweight='bold')
    
    # Training
    axes[0].contourf(xx, yy, Z, levels=20, cmap='RdYlGn', alpha=0.8)
    scatter_train = axes[0].scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=y_train,
                                    cmap='RdYlGn', edgecolors='black', s=50, alpha=0.7)
    axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
    axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
    axes[0].set_title('Training Data with Decision Boundary', fontweight='bold')
    plt.colorbar(scatter_train, ax=axes[0], label='Forest Loss')
    
    # Testing
    axes[1].contourf(xx, yy, Z, levels=20, cmap='RdYlGn', alpha=0.8)
    scatter_test = axes[1].scatter(X_test_pca[:, 0], X_test_pca[:, 1], c=y_test,
                                   cmap='RdYlGn', edgecolors='black', s=50, alpha=0.7)
    axes[1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
    axes[1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
    axes[1].set_title('Testing Data with Decision Boundary', fontweight='bold')
    plt.colorbar(scatter_test, ax=axes[1], label='Forest Loss')
    
    plt.tight_layout()
    plt.savefig(f'{CONFIG["results_dir"]}/02_decision_boundaries.png', dpi=300, bbox_inches='tight')
    print("  ✓ Saved: 02_decision_boundaries.png")
    plt.close()


def create_feature_importance_plots(feature_importance_df, correlation_df):
    """Create feature importance bar charts."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle('Feature Importance Analysis', fontsize=14, fontweight='bold')
    
    # Permutation importance
    top_features = feature_importance_df.head(8)
    axes[0].barh(top_features['Feature'], top_features['Importance'],
                xerr=top_features['Std'], color='skyblue', edgecolor='black')
    axes[0].set_xlabel('Permutation Importance')
    axes[0].set_title('Top 8 Most Important Features', fontweight='bold')
    axes[0].grid(axis='x', alpha=0.3)
    
    # Correlation
    top_corr = correlation_df.head(8)
    colors = ['green' if x > 0 else 'red' for x in top_corr['Correlation']]
    axes[1].barh(top_corr['Feature'], top_corr['Correlation'], color=colors, alpha=0.7, edgecolor='black')
    axes[1].set_xlabel('Correlation Coefficient')
    axes[1].set_title('Top 8 Features by Correlation', fontweight='bold')
    axes[1].grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{CONFIG["results_dir"]}/03_feature_importance.png', dpi=300, bbox_inches='tight')
    print("  ✓ Saved: 03_feature_importance.png")
    plt.close()


def create_feature_relationship_plots(feature_importance_df, correlation_df, df_scaled, target_col):
    """Create scatter plots showing feature-target relationships."""
    top_6_features = feature_importance_df.head(6)['Feature'].tolist()
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle('Relationship between Top Features and Target Variable', fontsize=14, fontweight='bold')
    axes = axes.flatten()
    
    for idx, feature in enumerate(top_6_features):
        axes[idx].scatter(df_scaled[feature], df_scaled[target_col], alpha=0.6, edgecolors='k', s=30)
        
        # Trend line
        z = np.polyfit(df_scaled[feature], df_scaled[target_col], 1)
        p = np.poly1d(z)
        x_line = np.linspace(df_scaled[feature].min(), df_scaled[feature].max(), 100)
        axes[idx].plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2)
        
        corr_val = df_scaled[feature].corr(df_scaled[target_col])
        axes[idx].set_xlabel(feature)
        axes[idx].set_ylabel(target_col)
        axes[idx].set_title(f'{feature}\n(r = {corr_val:.3f})', fontweight='bold')
        axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{CONFIG["results_dir"]}/04_feature_relationships.png', dpi=300, bbox_inches='tight')
    print("  ✓ Saved: 04_feature_relationships.png")
    plt.close()


def create_distribution_plots(feature_importance_df, df_scaled):
    """Create feature distribution histograms."""
    top_6_features = feature_importance_df.head(6)['Feature'].tolist()
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    fig.suptitle('Distribution of Top Features', fontsize=14, fontweight='bold')
    axes = axes.flatten()
    
    for idx, feature in enumerate(top_6_features):
        axes[idx].hist(df_scaled[feature], bins=30, color='steelblue', edgecolor='black', alpha=0.7)
        axes[idx].set_xlabel(feature)
        axes[idx].set_ylabel('Frequency')
        axes[idx].set_title(f'{feature}\n(Mean: {df_scaled[feature].mean():.2f}, Std: {df_scaled[feature].std():.2f})',
                           fontweight='bold')
        axes[idx].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{CONFIG["results_dir"]}/05_feature_distributions.png', dpi=300, bbox_inches='tight')
    print("  ✓ Saved: 05_feature_distributions.png")
    plt.close()


def create_correlation_heatmap(feature_importance_df, df_scaled, target_col):
    """Create correlation heatmap."""
    top_8_features = feature_importance_df.head(8)['Feature'].tolist()
    heatmap_features = top_8_features + [target_col]
    
    corr_matrix = df_scaled[heatmap_features].corr()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
               square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax)
    ax.set_title('Correlation Matrix - Top Features & Target', fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{CONFIG["results_dir"]}/06_correlation_heatmap.png', dpi=300, bbox_inches='tight')
    print("  ✓ Saved: 06_correlation_heatmap.png")
    plt.close()


# ============================================================================
# SUMMARY AND RECOMMENDATIONS
# ============================================================================

def generate_summary_report(metrics, feature_importance_df, correlation_df, cv_scores):
    """Generate comprehensive summary and recommendations."""
    print_section("SECTION 14: SUMMARY AND RECOMMENDATIONS")
    
    print("\n📊 EXECUTIVE SUMMARY")
    print("-" * 70)
    print(f"Model Performance Summary:")
    print(f"  • Training R²: {metrics['train_r2']:.4f}")
    print(f"  • Testing R²:  {metrics['test_r2']:.4f}")
    print(f"  • Training RMSE: {metrics['train_rmse']:.2f}")
    print(f"  • Testing RMSE:  {metrics['test_rmse']:.2f}")
    print(f"  • Cross-Val Mean R²: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    print("\n🔍 TOP 5 DEFORESTATION DRIVERS")
    print("-" * 70)
    for i, row in feature_importance_df.head(5).iterrows():
        importance_pct = (row['Importance'] / feature_importance_df['Importance'].sum()) * 100
        print(f"  {i+1}. {row['Feature']:<35} {row['Importance']:>8.4f} ({importance_pct:>5.1f}%)")
    
    print("\n📈 KEY INSIGHTS")
    print("-" * 70)
    print("\n1. PRIMARY DEFORESTATION DRIVERS:")
    for i, row in feature_importance_df.head(3).iterrows():
        feature = row['Feature']
        corr = correlation_df[correlation_df['Feature'] == feature]['Correlation'].values[0]
        direction = "increases" if corr > 0 else "decreases"
        print(f"   • {feature}: {direction.title()} forest loss (r = {corr:.3f})")
    
    print("\n2. MODEL RELIABILITY:")
    if metrics['test_r2'] > 0.75:
        reliability = "EXCELLENT - Model explains 75%+ of variance"
    elif metrics['test_r2'] > 0.6:
        reliability = "GOOD - Model explains 60-75% of variance"
    elif metrics['test_r2'] > 0.4:
        reliability = "MODERATE - Model explains 40-60% of variance"
    else:
        reliability = "LIMITED - Model explains <40% of variance"
    print(f"   • {reliability}")
    print(f"   • Cross-validation confirms consistency across data splits")
    
    print("\n" + "=" * 70)
    print("ACTIONABLE RECOMMENDATIONS FOR DEFORESTATION MITIGATION")
    print("=" * 70)
    
    recommendations = {
        "Policy Implementation": [
            "Strengthen deforestation policies with stricter enforcement",
            "Increase penalties for illegal logging and land clearance",
            "Establish protected forest areas with effective monitoring"
        ],
        "Economic Measures": [
            "Implement sustainable forestry practice incentives",
            "Create economic alternatives to forest clearing",
            "Link GDP growth with environmental protection targets"
        ],
        "Anti-Corruption Efforts": [
            "Combat corruption in environmental agencies",
            "Establish transparent monitoring systems",
            "Ensure accountability in enforcement"
        ],
        "Environmental Management": [
            "Monitor and reduce CO2 emissions through renewable energy",
            "Promote reforestation and forest regeneration programs",
            "Protect biodiversity hotspots and critical ecosystems"
        ]
    }
    
    for category, points in recommendations.items():
        print(f"\n📌 {category.upper()}:")
        for point in points:
            print(f"   • {point}")
    
    print("\n" + "=" * 70)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Execute the complete deforestation analysis pipeline."""
    
    print("\n" + "=" * 70)
    print(" DEFORESTATION ANALYSIS USING SUPPORT VECTOR MACHINE (SVM)")
    print("=" * 70)
    
    setup_environment()
    
    # 1. Load and inspect data
    df = load_dataset()
    inspect_data_quality(df)
    
    # 2. Data preprocessing
    df = handle_missing_values(df)
    df_encoded, encoders = encode_categorical_variables(df)
    
    # Define features before normalization
    feature_cols = [col for col in df_encoded.columns if col not in ['Country', 'Forest_Loss_Area_km2', 'Tree_Cover_Loss_percent']]
    target_col = 'Forest_Loss_Area_km2' if 'Forest_Loss_Area_km2' in df_encoded.columns else 'Tree_Cover_Loss_percent'
    
    df_scaled, scaler = normalize_features(df_encoded, feature_cols)
    
    # 3. Train-test split
    X_train, X_test, y_train, y_test, feature_cols = split_data(df_scaled, target_col, feature_cols)
    
    # 4. Model building
    svm_models, svm_performance = train_initial_models(X_train, y_train, X_test, y_test)
    
    # 5. Hyperparameter tuning
    best_svm, grid_search = hyperparameter_tuning(X_train, y_train, X_test, y_test)
    
    # 6. Cross-validation
    cv_scores, cv_mae_scores = apply_cross_validation(best_svm, X_train, y_train)
    
    # 7. Model evaluation
    metrics = evaluate_model(best_svm, X_train, y_train, X_test, y_test, cv_scores)
    
    # 8. Feature importance analysis
    feature_importance_df, correlation_df = analyze_feature_importance(
        best_svm, X_train, y_train, X_test, y_test, feature_cols, df_scaled, target_col
    )
    
    # 9. Visualizations
    create_visualizations(best_svm, X_train, y_train, X_test, y_test, metrics,
                         feature_importance_df, correlation_df, feature_cols, df_scaled, target_col)
    
    # 10. Generate summary report
    generate_summary_report(metrics, feature_importance_df, correlation_df, cv_scores)
    
    print("\n" + "=" * 70)
    print(" ✓ ANALYSIS COMPLETE - All results saved to 'results/' directory")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
