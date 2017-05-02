import pandas as pd
from pandas.tools.plotting import parallel_coordinates
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

df = pd.read_csv('../data/train.csv')
data = df[['Gender', 'Income', 'HouseholdStatus', 'EducationLevel', 'Party', 'Happy']].dropna()[:100]
print data


fig = plt.figure()
parallel_coordinates(data,'Happy')
fig.savefig("paral_coord.jpg")
