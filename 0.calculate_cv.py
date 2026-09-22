import pandas as pd
import numpy as np

# Input file path
input_file = "因子分布.xlsx"

# Output file path
output_file = "因子CV结果.xlsx"

# Read data
df = pd.read_excel(input_file)

# Factor columns for CV calculation
factor_cols = [
    "si10", "sp", "ssr", "t2m", "tp",
    "Lake_area", "Shore_dev", "Dis_avg", "Res_time", "Elevation",
    "Cropland", "Forest", "Grassland", "Urban", "Barren"
]

# Check whether all required columns exist
missing_cols = [col for col in factor_cols if col not in df.columns]
if missing_cols:
    raise ValueError(f"The following columns are missing from the table: {missing_cols}")

# Convert to numeric values; values that cannot be converted are set to NaN
data = df[factor_cols].apply(pd.to_numeric, errors="coerce")

# Calculate statistics
result = pd.DataFrame({
    "Factor": factor_cols,
    "N": data[factor_cols].count().values,
    "Mean": data[factor_cols].mean().values,
    "SD": data[factor_cols].std(ddof=1).values,
    "CV": (data[factor_cols].std(ddof=1) / data[factor_cols].mean()).values,
    "CV_percent": (data[factor_cols].std(ddof=1) / data[factor_cols].mean() * 100).values,
    "Min": data[factor_cols].min().values,
    "Median": data[factor_cols].median().values,
    "Max": data[factor_cols].max().values,
    "Missing_N": data[factor_cols].isna().sum().values
})

# If the mean is 0, CV is undefined and is set to NaN
result.loc[result["Mean"] == 0, ["CV", "CV_percent"]] = np.nan

# Sort by CV from largest to smallest
result = result.sort_values("CV_percent", ascending=False)

# Save results
result.to_excel(output_file, index=False)

print("CV calculation completed!")
print(f"Results have been saved to: {output_file}")
print(result)