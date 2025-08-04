


# Supply Chain Emissions Modeling

## Table of Contents
- [Project Overview](#project-overview)
- [Background & Motivation](#background--motivation)
- [Data Description](#data-description)
- [Workflow & Methodology](#workflow--methodology)
- [Modeling Approach](#modeling-approach)
- [Evaluation Metrics](#evaluation-metrics)
- [Setup & Installation](#setup--installation)
- [Usage Instructions](#usage-instructions)
- [Customization & Extensions](#customization--extensions)
- [Project Structure](#project-structure)
- [Troubleshooting](#troubleshooting)
- [License](#license)
- [Contact](#contact)

## Project Overview
This project predicts supply chain emission factors for US industries and commodities (2010–2016) using advanced machine learning. It leverages descriptive and quality metrics to build a robust regression model for estimating emission factors with associated margins, supporting sustainability analysis and decision-making.

## Background & Motivation
Supply chain emissions are a critical component of environmental impact assessments. Accurate prediction of emission factors helps industries, policymakers, and researchers understand and reduce greenhouse gas emissions. This project automates the process using historical data and state-of-the-art ML techniques.

## Data Description
- **Source:** Excel file with multiple sheets, each representing a year (2010–2016) and data type (Commodity or Industry).
- **Columns:**
  - `Code`, `Name`: Identifiers for commodity/industry
  - `Substance`: Type of greenhouse gas (e.g., CO2, methane)
  - `Unit`: Measurement unit
  - `Margins`, `Reliability`, `Correlations`, etc.: Quality and descriptive metrics
  - `Supply Chain Emission Factors with Margins`: Target variable

## Workflow & Methodology
1. **Import Libraries:** Load all required Python libraries for data analysis, visualization, and modeling.
2. **Load Dataset:** Read all relevant sheets from the Excel file for each year, combine commodity and industry data, and standardize columns.
3. **Data Preprocessing:**
   - Drop unnecessary columns (e.g., unnamed, Name, Code, Year)
   - Encode categorical variables (Substance, Unit, Source)
   - Handle missing values and ensure all features are numeric
   - Visualize distributions and check for data quality issues
4. **Feature Engineering:**
   - Define features (`X`) and target (`y`)
   - Normalize features using `StandardScaler`
5. **Train-Test Split:** Split the data into training and testing sets (typically 80/20 split)
6. **Model Selection & Training:**
   - Use `XGBRegressor` (XGBoost) for efficient and accurate regression
   - Fit the model on the training data
7. **Evaluation:**
   - Predict on the test set
   - Calculate RMSE and R² to assess model performance
8. **Hyperparameter Tuning:**
   - Use `GridSearchCV` to find the best hyperparameters for XGBoost
   - Re-train the model with the best parameters and evaluate again
9. **Model Saving:** Save the trained model and scaler using `joblib`

## Modeling Approach
- **Algorithm:** XGBoost (Extreme Gradient Boosting) is used for its speed and accuracy on tabular data.
- **Why XGBoost?**
  - Handles missing values and categorical features efficiently
  - Often outperforms other algorithms for structured data
- **Alternatives:** LightGBM, CatBoost, Random Forest (can be tried for comparison)

## Evaluation Metrics
- **RMSE (Root Mean Squared Error):** Measures average prediction error magnitude
- **R² Score (Coefficient of Determination):** Indicates proportion of variance explained by the model

## Setup & Installation
1. Clone or download this repository.
2. Place the Excel data file in the project directory.
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   Or install manually:
   ```bash
   pip install pandas numpy seaborn matplotlib scikit-learn xgboost joblib
   ```

## Usage Instructions
1. Open `experiment.ipynb` in Jupyter or VS Code.
2. Run all cells in order.
3. The final model and scaler will be saved in the `models/` directory.
4. Review the output for model performance metrics.

## Customization & Extensions
- Try other algorithms (LightGBM, CatBoost) for comparison
- Adjust feature engineering and hyperparameters as needed
- Add new features or use domain knowledge to improve accuracy
- Use k-fold cross-validation for more robust evaluation

## Project Structure
- `experiment.ipynb` — Main notebook with code and workflow
- `project_explanation.md` — Detailed project explanation
- `models/` — Saved model and scaler
- `SupplyChainEmissionFactorsforUSIndustriesCommodities (1).xlsx` — Data file

## Troubleshooting
- **Data Not Loading:** Ensure the Excel file path is correct and all required sheets are present.
- **Module Not Found:** Double-check that all dependencies are installed.
- **Low Accuracy:**
  - Check for data quality issues or missing values
  - Try more advanced feature engineering or hyperparameter tuning
  - Experiment with alternative algorithms

## License
This project is for educational and research purposes.

## Contact
For questions or collaboration, please contact jbarathvishnu2005@gmail.com



# GitHub Repository

Project link: [https://github.com/BARATHVISHNU-J/GreenHouse_Gas_Prediction.git](https://github.com/BARATHVISHNU-J/GreenHouse_Gas_Prediction.git)


