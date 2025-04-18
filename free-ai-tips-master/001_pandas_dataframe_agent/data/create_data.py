import pandas as pd
import numpy as np

# Parameters
num_transactions = 100
start_date = '2023-01-01'

# Generate Transaction IDs
transaction_ids = np.arange(1, num_transactions + 1)

# Generate Dates
dates = pd.date_range(start=start_date, periods=num_transactions, freq='D')

# Generate Customer IDs
customer_ids = np.random.choice(['C{:03d}'.format(i) for i in range(1, 31)], num_transactions)

# Generate Products
products = np.random.choice(['Product A', 'Product B', 'Product C', 'Product D', 'Product E'], num_transactions)

# Generate Quantities
quantities = np.random.randint(1, 10, num_transactions)

# Generate Prices
price_dict = {'Product A': 15.00, 'Product B': 20.00, 'Product C': 50.00, 'Product D': 25.00, 'Product E': 30.00}
prices = [price_dict[product] for product in products]

# Generate Geographies
geographies = np.random.choice(['North', 'South', 'East', 'West'], num_transactions)

# Calculate Sales
sales = quantities * prices

# Create DataFrame
data = {
    'TransactionID': transaction_ids,
    'Date': dates,
    'CustomerID': customer_ids,
    'Product': products,
    'Quantity': quantities,
    'Price': prices,
    'Sales': sales,
    'Geography': geographies
}

df = pd.DataFrame(data)

# Save to CSV
df.to_csv('001_pandas_dataframe_agent/data/customer_data.csv', index=False)