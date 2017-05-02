import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('../data/train.csv')
income_and_happiness = df[['Income', 'Happy']].dropna()
income_and_happiness = np.array(income_and_happiness)

n_group = 6
income_tuple = []
income_tuple.append(np.where(income_and_happiness == "under $25,000")[0])
income_tuple.append(np.where(income_and_happiness == "$25,001 - $50,000")[0])
income_tuple.append(np.where(income_and_happiness == "$50,000 - $74,999")[0])
income_tuple.append(np.where(income_and_happiness == "$75,000 - $100,000")[0])
income_tuple.append(np.where(income_and_happiness == "$100,001 - $150,000")[0])
income_tuple.append(np.where(income_and_happiness == "over $150,000")[0])


happy_idx = np.where(income_and_happiness  == 1)[0]
unhappy_idx = np.where(income_and_happiness  == 0)[0]

happy_income_count = []
unhappy_income_count = []
for i in range(6):
	happy_income_count.append(np.intersect1d(income_tuple[i],happy_idx).shape[0])
	unhappy_income_count.append(np.intersect1d(income_tuple[i],unhappy_idx).shape[0])

total_count = []
for i in range(6):
	total_count.append(happy_income_count[i]+unhappy_income_count[i])

for i in range(6):
	happy_income_count[i]=float(happy_income_count[i])/total_count[i]
	unhappy_income_count[i] = float(unhappy_income_count[i])/total_count[i]


bar_width = 0.35
index = np.arange(n_group)
rect1 = plt.bar(index, happy_income_count, bar_width, color='b', label="happy")
rect2 = plt.bar(index+bar_width, unhappy_income_count, bar_width, color='c', label="unhappy")
plt.title("Bar Chart of Income and Happiness")
plt.xlabel("Income Levels")
plt.ylabel("Fraction of Happy/Unhappy")
plt.xticks(index + bar_width, ("under $25,000", "$25,001 - $50,000", "$50,000 - $74,999", 
							"$75,000 - $100,000", "$100,001 - $150,000", "over $150,000"), rotation = 11)
plt.legend()
# attach some text labels
count = 0
for rect in rect1:
    height = rect.get_height()
    plt.text(rect.get_x() + rect.get_width()/2., height, "%.2f" %happy_income_count[count],ha='center', va='bottom')
    count += 1

count = 0
for rect in rect2:
    height = rect.get_height()
    plt.text(rect.get_x() + rect.get_width()/2., height, "%.2f" %unhappy_income_count[count],ha='center', va='bottom')
    count += 1

plt.tight_layout()
plt.savefig("bar_chart.png")

