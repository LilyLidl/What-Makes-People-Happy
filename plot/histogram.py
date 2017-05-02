#!/usr/bin/env python
import pandas as pd
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt


df = pd.read_csv('../data/train.csv')
YOB = df['YOB'].dropna().to_dict().values()
YOB = np.array(YOB).astype(int)
print YOB
plt.hist(df['YOB'], normed=1)
plt.show()
