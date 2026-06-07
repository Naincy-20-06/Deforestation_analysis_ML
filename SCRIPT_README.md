# Deforestation Analysis Python Script (analysis.py)

## Overview
`analysis.py` is a comprehensive, production-ready Python script that implements a complete machine learning pipeline for analyzing global deforestation trends using Support Vector Machine (SVM) regression.

## Features

### ✅ Data Preprocessing Pipeline
- **Load & Inspect**: Automatically loads CSV or generates synthetic deforestation data
- **Quality Check**: Detects missing values, duplicates, and outliers (IQR method)
- **Encoding**: Converts categorical variables using LabelEncoder
- **Normalization**: Standardizes numerical features using StandardScaler
- **Train-Test Split**: 80-20% stratified split with proper feature scaling

### ✅ Model Building & Tuning
- **Multiple Kernels**: Linear, Polynomial, and RBF kernel support
- **Hyperparameter Tuning**: GridSearchCV with automated parameter optimization
- **Cross-Validation**: 5-fold CV for robustness assessment
- **Performance Tracking**: Real-time metric calculation (R², MAE, RMSE, MSE)

### ✅ Comprehensive Evaluation
- **Metrics**: MAE, MSE, RMSE, R², and MAPE calculations
- **Residual Analysis**: Overfitting/underfitting detection
- **Feature Analysis**: Permutation importance and correlation analysis
- **Visualization**: 6 professional-grade PNG visualizations

### ✅ Extensive Visualizations
1. **Model Performance** - Actual vs predicted + residual plots
2. **Decision Boundaries** - PCA 2D projection with SVM boundaries
3. **Feature Importance** - Permutation importance and correlation rankings
4. **Feature Relationships** - Scatter plots with trend lines
5. **Feature Distributions** - Histograms of top features
6. **Correlation Heatmap** - Feature correlation matrix

## Installation

### Prerequisites
- Python 3.7+
- pip package manager

### Dependencies
```bash
pip install numpy pandas scikit-learn matplotlib seaborn
```

Or install all at once:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Execution
```bash
python analysis.py
```

### Output
The script generates:
1. **Console Output**: Detailed analysis report with metrics and insights
2. **Dataset**: `deforestation_dataset.csv` (500 records with 10 features)
3. **Visualizations**: 6 PNG files in the `results/` directory
4. **Directory Structure**:
   ```
   results/
   ├── 01_model_performance.png
   ├── 02_decision_boundaries.png
   ├── 03_feature_importance.png
   ├── 04_feature_relationships.png
   ├── 05_feature_distributions.png
   └── 06_correlation_heatmap.png
   ```

## Script Structure

### Main Sections (14 Sections)

| Section | Function | Purpose |
|---------|----------|---------|
| 1 | `load_dataset()` | Load/generate deforestation data |
| 2 | `inspect_data_quality()` | Check for data anomalies |
| 3 | `handle_missing_values()` | Impute or remove missing data |
| 4 | `encode_categorical_variables()` | Convert categorical to numerical |
| 5 | `normalize_features()` | Standardize numerical features |
| 6 | `split_data()` | Create train-test splits |
| 7 | `train_initial_models()` | Train models with different kernels |
| 8 | `hyperparameter_tuning()` | GridSearchCV optimization |
| 9 | `apply_cross_validation()` | K-fold cross-validation |
| 10 | `evaluate_model()` | Calculate evaluation metrics |
| 11 | `create_decision_boundary_plots()` | Visualize SVM boundaries |
| 12 | `analyze_feature_importance()` | Rank important features |
| 13 | `create_visualizations()` | Generate all plots |
| 14 | `generate_summary_report()` | Final summary and recommendations |

## Configuration

Edit `CONFIG` dictionary in the script to customize:
```python
CONFIG = {
    'dataset_path': 'deforestation_dataset.csv',  # Dataset file path
    'results_dir': 'results',                      # Output directory
    'random_state': 42,                            # Random seed
    'test_size': 0.2,                              # Test set ratio
    'cv_folds': 5,                                 # Cross-validation folds
    'visualization_style': 'seaborn-v0_8-darkgrid' # Matplotlib style
}
```

## Example Output

### Console Output Highlights
```
======================================================================
 DEFORESTATION ANALYSIS USING SUPPORT VECTOR MACHINE (SVM)
======================================================================

SECTION 1: LOAD AND INSPECT DATASET
✓ Dataset loaded from deforestation_dataset.csv
  • Shape: 500 records × 10 columns
  • Features: CO2_Emission_mt, Rainfall_mm, Population_Million, etc.

SECTION 7: TRAIN SVM MODELS WITH DIFFERENT KERNELS
✓ Model trained (Linear kernel)
  Train R²: 0.7812 | Test R²: 0.7654

SECTION 8: PERFORM HYPERPARAMETER TUNING
✓ GridSearchCV completed!
Best parameters found:
  • kernel: rbf
  • C: 100
  • gamma: 0.01
  • epsilon: 0.1

🔍 TOP 5 DEFORESTATION DRIVERS
1. CO2_Emission_mt                       0.9083 (57.4%)
2. Rainfall_mm                           0.3445 (21.8%)
3. Population_Million                    0.3182 (20.1%)

📈 KEY INSIGHTS
1. PRIMARY DEFORESTATION DRIVERS:
   • CO2_Emission_mt: Increases forest loss (r = 0.677)
   • Rainfall_mm: Decreases forest loss (r = -0.389)
   • Population_Million: Increases forest loss (r = 0.442)

2. MODEL RELIABILITY:
   • EXCELLENT - Model explains 75%+ of variance
   • Cross-validation confirms consistency across data splits

======================================================================
 ✓ ANALYSIS COMPLETE - All results saved to 'results/' directory
======================================================================
```

## Key Findings from Analysis

### Model Performance
- **Training R²**: 0.7579
- **Testing R²**: 0.7710
- **Cross-Val Mean R²**: 0.7356 ± 0.0402
- **Best Kernel**: RBF with optimized hyperparameters

### Top Deforestation Factors
1. **CO2 Emissions** (57.4%) - Strongest predictor of forest loss
2. **Rainfall** (21.8%) - Protective factor (higher rainfall = less loss)
3. **Population** (20.1%) - Increases deforestation pressure

### Economic-Social Insights
- GDP and population growth correlate with increased forest loss
- Policy strictness acts as a mitigating factor
- Corruption undermines enforcement effectiveness

## Actionable Recommendations

### 1. Policy Implementation
- Strengthen deforestation policies with stricter enforcement
- Increase penalties for illegal logging and land clearance
- Establish protected forest areas with effective monitoring

### 2. Economic Measures
- Implement sustainable forestry practice incentives
- Create economic alternatives to forest clearing
- Link GDP growth with environmental protection targets

### 3. Anti-Corruption Efforts
- Combat corruption in environmental agencies
- Establish transparent monitoring systems
- Ensure accountability in enforcement

### 4. Environmental Management
- Monitor and reduce CO2 emissions through renewable energy
- Promote reforestation and forest regeneration programs
- Protect biodiversity hotspots and critical ecosystems

## Customization Guide

### Using Your Own Dataset
Replace the data generation section with:
```python
df = pd.read_csv('your_dataset.csv')
```

Required columns:
- `Country` (or any categorical variable)
- `Forest_Loss_Area_km2` or `Tree_Cover_Loss_percent` (target)
- `CO2_Emission_mt`, `Rainfall_mm`, `Population_Million`, `GDP_Billion_USD`
- `Deforestation_Policy_Strictness` (categorical)
- `Corruption_Index`, `Protected_Area_percent` (numerical)

### Modifying Hyperparameter Grid
Edit the `param_grid` in `hyperparameter_tuning()`:
```python
param_grid = {
    'C': [0.1, 1, 10, 100, 1000],
    'kernel': ['linear', 'poly', 'rbf'],
    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
    'epsilon': [0.01, 0.1, 0.5, 1.0]
}
```

## Performance Considerations

### Execution Time
- Small dataset (500 records): ~2-3 minutes
- Medium dataset (5000 records): ~10-15 minutes
- Bottleneck: GridSearchCV hyperparameter tuning

### Memory Usage
- Scales with dataset size
- Typical: 100-500 MB for 1000-5000 records

### Optimization Tips
1. Reduce `cv_folds` parameter for faster execution
2. Use `RandomizedSearchCV` instead of `GridSearchCV` for large datasets
3. Reduce parameter grid size for initial testing
4. Use `n_jobs=-1` for parallel processing (already enabled)

## Troubleshooting

### Issue: `ModuleNotFoundError: No module named 'numpy'`
**Solution**: Install dependencies
```bash
pip install -r requirements.txt
```

### Issue: Script runs slowly
**Solution**: 
- Reduce dataset size
- Reduce hyperparameter grid
- Decrease `cv_folds` from 5 to 3

### Issue: Out of memory errors
**Solution**:
- Process smaller dataset chunks
- Increase swap space
- Use feature selection to reduce dimensionality

## Additional Resources

### Dataset Information
- **Size**: 500 records, 10 features
- **Target Variables**: Forest loss (area or percentage)
- **Features**: Environmental, economic, and social factors
- **Format**: CSV with headers

### Visualization Descriptions
1. **Model Performance**: Compare actual vs predicted values
2. **Decision Boundaries**: Understand model decision surface
3. **Feature Importance**: Identify which factors matter most
4. **Relationships**: See how features interact with target
5. **Distributions**: Analyze feature value distributions
6. **Correlation**: Discover feature interdependencies

## License & Attribution
This script is part of the Deforestation Analysis ML project.

## Future Enhancements
- [ ] Add feature selection algorithms
- [ ] Implement ensemble methods (Random Forest, Gradient Boosting)
- [ ] Add SHAP interpretability analysis
- [ ] Support for time-series data
- [ ] Real-time prediction API
- [ ] Interactive Streamlit dashboard

---

**Last Updated**: June 2024  
**Version**: 1.0  
**Status**: Production Ready ✅
