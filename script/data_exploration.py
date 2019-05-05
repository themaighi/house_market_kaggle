import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import os
## Loading the data

house_price_dt = pd.read_csv('data/train.csv')
house_price_dt['log_price'] = np.log(house_price_dt.SalePrice)

# Check the distribution of the y variable

N_points = 100000
n_bins = 20

fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)

axs[0].hist(house_price_dt.SalePrice, bins=n_bins)
axs[1].hist(house_price_dt.log_price, bins=n_bins)
axs[0].set_title('House Price')
axs[1].set_title('Log House Price')

plt.savefig('distribution.png')

# describe the dataset


house_price_dt.describe()

## Check the missing values (Delete missing values variable)

## Create a correlation matrix

# subset of the object and of the numerical data

dtypes_columns = pd.DataFrame({'type': house_price_dt.dtypes})
numerical_values = dtypes_columns[(dtypes_columns.type == 'int64')|(dtypes_columns.type == 'float64')]
col_for_correlation = [col for col in numerical_values.index if col not in ['Id', 'SalePrice']]

correlation_dataset = house_price_dt[col_for_correlation].corr()
pd.plotting.scatter_matrix(house_price_dt[col_for_correlation], figsize=(6, 6))

plt.savefig('correlation.png')


# Heatmap

plt.imshow(correlation_dataset, cmap='hot', interpolation='nearest')
plt.savefig('heatmap.png')
## Visually check the direction of the relation
## Run an easy model as benchmark
## Use light GBM to get some insight
## Use partial dependency plot to how variables respond to each other
## Write conclusion



