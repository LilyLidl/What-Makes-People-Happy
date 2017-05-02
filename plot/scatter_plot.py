import pandas as pd
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

df = pd.read_csv('../data/train.csv')
income = df['Income']
year = df['YOB']


# Do value conversion on income:
for i in range(len(income)):
    if income[i] == 'under $25,000':
        income[i] = 12500
    elif income[i] == '$25,001 - $50,000':
        income[i] = 37500
    elif income[i] == '$50,000 - $74,999':
        income[i] = 62500
    elif income[i] == '$75,000 - $100,000':
        income[i] = 87500
    elif income[i] == '$100,001 - $150,000':
        income[i] = 125000
    elif income[i] == 'over $150,000':
        income[i] = 187500

year_max = max(year)
year_min = min(year)

income_idx = np.where(np.logical_not(np.isnan(income)))[0]
year_idx = np.where(np.logical_not(np.isnan(year)))[0]
valid_idx = np.intersect1d(income_idx,year_idx)
income = np.array(income)
year = np.array(year)


fig = plt.figure()
fig.suptitle('Scatter plot of YOB and income')
plt.xlabel('Year of birth')
plt.ylabel('Income')
plt.scatter(year[valid_idx],income[valid_idx],alpha=0.2)
fig.savefig('scatter_plot.jpg')
