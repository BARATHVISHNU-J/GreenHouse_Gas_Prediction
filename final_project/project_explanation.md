# Supply Chain Emissions Modeling Project: Full Explanation

## Project Overview
This project aims to predict supply chain emission factors for US industries and commodities from 2010–2016. The goal is to use descriptive and quality metrics to build a regression model that can estimate emission factors with associated margins.

## Data Description
- **Source:** Excel file with multiple sheets, each representing a year and data type (Commodity or Industry).
- **Columns:**
  - Code, Name: Identifiers for commodity/industry
  - Substance: Type of greenhouse gas (e.g., CO2, methane)
  - Unit: Measurement unit
  - Margins, Reliability, Correlations, etc.: Quality and descriptive metrics
  - Supply Chain Emission Factors with Margins: Target variable

## Workflow Steps

### 1. Import Required Libraries
- `pandas`, `numpy`: Data manipulation
- `seaborn`, `matplotlib`: Visualization
- `sklearn`: Preprocessing, model selection, metrics
- `xgboost`: Efficient regression model
- `joblib`: Model saving

### 2. Load Dataset
- Read all relevant sheets from the Excel file for each year (2010–2016).
- Combine commodity and industry data, add a 'Source' column, and standardize column names.
- Concatenate all years into a single DataFrame.

### 3. Data Preprocessing
- Drop unnecessary columns (e.g., unnamed, Name, Code, Year).
- Encode categorical variables (Substance, Unit, Source) using mapping dictionaries.
- Handle missing values and ensure all features are numeric.
- Visualize distributions and check for data quality issues.

### 4. Feature Engineering
- Define features (`X`) and target (`y`).
- Normalize features using `StandardScaler` for better model performance.

### 5. Train-Test Split
- Split the data into training and testing sets (typically 80/20 split).

### 6. Model Selection and Training
- Use `XGBRegressor` (from XGBoost) for efficient and accurate regression.
- Fit the model on the training data.

### 7. Evaluation
- Predict on the test set.
- Calculate RMSE (Root Mean Squared Error) and R² (coefficient of determination) to assess model performance.

### 8. Hyperparameter Tuning
- Use `GridSearchCV` to find the best hyperparameters for XGBoost (e.g., n_estimators, max_depth, learning_rate, subsample).
- Re-train the model with the best parameters and evaluate again.

### 9. Model Saving
- Save the trained model and scaler using `joblib` for future use.

## Key Points
- **Why XGBoost?**
  - Faster and often more accurate than Random Forest for tabular data.
  - Handles missing values and categorical features efficiently.
- **Data Quality:**
  - Careful preprocessing and encoding are crucial for model accuracy.
- **Reproducibility:**
  - All steps are scripted in the notebook for easy reruns and modifications.

## How to Use
1. Run the notebook cells in order.
2. Ensure the Excel data file is in the correct path.
3. The final model and scaler will be saved in the `models/` directory.

## Customization
- You can try other algorithms (e.g., LightGBM, CatBoost) for comparison.
- Adjust feature engineering and hyperparameters as needed for your specific data.

---

This file provides a comprehensive explanation of each part of the project. For code details, refer to the notebook `experiment.ipynb`.
