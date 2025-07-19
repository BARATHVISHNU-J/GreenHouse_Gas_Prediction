import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# 1. Load datasets
file_2015 = 'SupplyChainEmissionFactorsforUSIndustriesCommodities(2015_Summary_Commodity).csv'
file_2016 = 'SupplyChainEmissionFactorsforUSIndustriesCommodities(2016_Summary_Commodity).csv'
lcia_file = 'SupplyChainEmissionFactorsforUSIndustriesCommodities(LCIA Factors of Other GHGs).csv'

df_2015 = pd.read_csv(file_2015)
df_2016 = pd.read_csv(file_2016)
df_lcia = pd.read_csv(lcia_file)

# 2. Basic cleaning and preprocessing
# Standardize column names (remove extra spaces, lower case)
df_2015.columns = df_2015.columns.str.strip().str.lower().str.replace(' ', '_')
df_2016.columns = df_2016.columns.str.strip().str.lower().str.replace(' ', '_')
df_lcia.columns = df_lcia.columns.str.strip().str.lower().str.replace(' ', '_')

# For demonstration, use 2016 data (can be extended to merge with 2015)
df = df_2016.copy()

# Drop unnamed/empty columns if present
df = df.loc[:, ~df.columns.str.contains('^unnamed')]

# Check for missing values
df = df.dropna(subset=['supply_chain_emission_factors_with_margins'])

# Convert numeric columns to float
df['supply_chain_emission_factors_with_margins'] = pd.to_numeric(df['supply_chain_emission_factors_with_margins'], errors='coerce')

# 3. Feature engineering: Pivot so each (commodity_code, commodity_name) is a row, GHGs as columns
pivot = df.pivot_table(index=['commodity_code', 'commodity_name'],
                      columns='substance',
                      values='supply_chain_emission_factors_with_margins').reset_index()

# Fill missing values with 0 (or could use mean/median)
pivot = pivot.fillna(0)

# 4. Feature exploration: Correlation heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(pivot.select_dtypes(include=[np.number]).corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap of GHG Emission Factors')
plt.show()

# 5. Encode categorical features
le_code = LabelEncoder()
le_name = LabelEncoder()
pivot['commodity_code_enc'] = le_code.fit_transform(pivot['commodity_code'])
pivot['commodity_name_enc'] = le_name.fit_transform(pivot['commodity_name'])

# 6. Prepare features and target
# Example: Predict total GHG (sum of all gases) from code/name
pivot['total_ghg'] = pivot[['carbon dioxide', 'methane', 'nitrous oxide', 'other ghgs']].sum(axis=1)
X = pivot[['commodity_code_enc', 'commodity_name_enc']]
y = pivot['total_ghg']

# 7. Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 8. Baseline model: RandomForestRegressor (can swap for LinearRegression)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 9. Predict and evaluate
y_pred = model.predict(X_test)
print('R^2 Score:', r2_score(y_test, y_pred))
print('MAE:', mean_absolute_error(y_test, y_pred))
print('MSE:', mean_squared_error(y_test, y_pred))

# 10. Feature importance plot
plt.figure(figsize=(6, 4))
feat_imp = pd.Series(model.feature_importances_, index=X.columns)
feat_imp.sort_values().plot(kind='barh')
plt.title('Feature Importances')
plt.show()

# ---
# Next steps for improvement:
# - Merge 2015 and 2016 for time-based features
# - Use LCIA factors for more detailed GHG breakdown
# - Add more features (e.g., DQ scores)
# - Hyperparameter tuning, cross-validation
# - Try other models (XGBoost, etc.)
# - Save model for deployment 