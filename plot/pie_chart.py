import pandas as pd
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

df = pd.read_csv('../data/train.csv')
gender = np.array(df['Gender'].to_dict().values())
happy = np.array(df['Happy'].to_dict().values())



# Count the number of occurence:

male_idx = np.where(gender == 'Male')[0]
female_idx = np.where(gender == 'Female')[0]
happy_idx = np.where(happy == 1)[0]
unhappy_idx = np.where(happy == 0)[0]

happy_male_count = np.intersect1d(male_idx,happy_idx).shape[0]
unhappy_male_count = np.intersect1d(male_idx,unhappy_idx).shape[0]
happy_female_count = np.intersect1d(female_idx,happy_idx).shape[0]
unhappy_female_count = np.intersect1d(female_idx,unhappy_idx).shape[0]



# Draw figure:

f, (ax1,ax2) = plt.subplots(1,2)
f.set_size_inches(10,5,forward=True)
f.suptitle('Pie chart for the fraction of happy men/women')
explode = (0, 0.1) # "explode" the "unhappy" slice
ax1.set_title('Men')
ax1.pie([happy_male_count,unhappy_male_count], explode=explode, labels=['Happy', 'Unhappy'], autopct='%1.1f%%', shadow=True, startangle=90)
ax2.set_title('Women')
ax2.pie([happy_female_count,unhappy_female_count], explode=explode, labels=['Happy', 'Unhappy'], autopct='%1.1f%%', shadow=True, startangle=90)
f.savefig('pie_chart.jpg')
