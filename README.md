# Greenhouse Gas Emission Prediction for US Industries and Commodities

This project provides a baseline predictive model to estimate greenhouse gas (GHG) emissions based on US industry and commodity data. It uses supply chain emission factor datasets for 2015 and 2016, as well as LCIA factors for other GHGs.

## Datasets
- **SupplyChainEmissionFactorsforUSIndustriesCommodities(2015_Summary_Commodity).csv**: Emission factors for 2015 by commodity/industry and GHG type.
- **SupplyChainEmissionFactorsforUSIndustriesCommodities(2016_Summary_Commodity).csv**: Emission factors for 2016 by commodity/industry and GHG type.
- **SupplyChainEmissionFactorsforUSIndustriesCommodities(LCIA Factors of Other GHGs).csv**: Global warming potential (GWP-100) factors for various other GHGs.
- **SupplyChainEmissionFactorsforUSIndustriesCommodities(Data Dictionary).csv**: Definitions and explanations for all columns in the datasets.

## Model Script: `ghg_prediction_model.py`
This script demonstrates a full workflow for building a GHG emission prediction model:

1. **Data Loading**: Reads the 2015, 2016, and LCIA datasets using pandas.
2. **Cleaning & Preprocessing**:
   - Standardizes column names.
   - Handles missing values and data types.
   - Pivots the data so each row is a unique (industry, commodity) with GHGs as columns.
3. **Feature Exploration**:
   - Visualizes feature correlations using a heatmap (matplotlib/seaborn).
4. **Encoding**:
   - Encodes categorical features (industry/commodity) using label encoding.
5. **Train/Test Split**: Splits the data into training and test sets.
6. **Modeling**:
   - Trains a baseline RandomForestRegressor to predict total GHG emissions.
   - Evaluates the model using R², MAE, and MSE.
   - Plots feature importances.

## How to Run
1. Install dependencies:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn
   ```
2. Run the script:
   ```bash
   python ghg_prediction_model.py
   ```

## Next Steps & Improvements
- Merge 2015 and 2016 data for time-based features.
- Use LCIA factors for more detailed GHG breakdown.
- Add more features (e.g., data quality scores).
- Hyperparameter tuning, cross-validation, and advanced models (e.g., XGBoost).
- Save and deploy the trained model.

## License
This project is for educational and research purposes. 