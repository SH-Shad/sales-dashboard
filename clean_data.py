import pandas as pd

# Load the dataset
df = pd.read_csv("superstore.csv", encoding="latin-1")

# Fix date columns
df['Order Date'] = pd.to_datetime(df['Order Date'])
df['Ship Date'] = pd.to_datetime(df['Ship Date'])

# Create new useful columns
df['Month'] = df['Order Date'].dt.to_period('M').astype(str)
df['Year'] = df['Order Date'].dt.year
df['Profit Margin'] = (df['Profit'] / df['Sales']).round(4)

# Remove duplicates and nulls
df.drop_duplicates(inplace=True)
df.dropna(inplace=True)

# Save cleaned data
df.to_csv("cleaned_superstore.csv", index=False)

print("✅ Data cleaned! Rows:", len(df))
print(df.head())