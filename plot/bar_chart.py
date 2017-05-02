import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('../data/train.csv')
income_and_happiness = df[['Income', 'Happy']].dropna()
income_and_happiness = np.array(income_and_happiness)

n_group = 6
income1_tuple = np.where(income_and_happiness == "under $25,000")[0]
income2_tuple = np.where(income_and_happiness == "$25,001 - $50,000")[0]
income3_tuple = np.where(income_and_happiness == "$50,000 - $74,999")[0]
income4_tuple = np.where(income_and_happiness == "$75,000 - $100,000")[0]
income5_tuple = np.where(income_and_happiness == "$100,001 - $150,000")[0]
income6_tuple = np.where(income_and_happiness == "over $150,000")[0]

happy_idx = np.where(income_and_happiness  == 1)[0]
unhappy_idx = np.where(income_and_happiness  == 0)[0]

