
#!/usr/bin/env python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df = pd.read_csv('../data/train.csv')
YOB = df['YOB'].dropna()
YOB = np.array(YOB).astype(int)
uniqueYOB = np.unique(YOB) # 70
minYOB = np.amin(uniqueYOB)
maxYOB = np.amax(uniqueYOB)
binsize = 15

plt.hist(YOB, normed=1, bins=binsize, ec='black')
axes = plt.gca()
plt.title("Histogram of YOB")
plt.xlabel("YOB (year of birth)")
plt.ylabel("Probability Density of YOB")

plt.savefig("yob_histogram.png")


