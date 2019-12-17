import pandas as pd
import numpy as np

from sklearn.metrics import f1_score, roc_auc_score, accuracy_score

from matplotlib import pyplot as plt
import seaborn as sns

# some hacky analysis from yours truly

df = pd.read_csv('data/threshold_results_shap.csv', index_col=0)

f1s, fsts, scnds, thrds = [], [], [], []

for trial in np.unique(df['trial']):
	relevant_runs = df[df.trial == trial]

	yhat = relevant_runs['yhat']
	y = relevant_runs['y']

	# need to flip classes
	yhat = 1 - yhat
	y = 1 - y

	pct_first = relevant_runs['pct_occur_first'].values[0]
	pct_second = relevant_runs['pct_occur_second'].values[0]
	pct_third = relevant_runs['pct_occur_third'].values[0]

	f1 = roc_auc_score(y, yhat)

	f1s.append(f1)
	fsts.append(pct_first)
	scnds.append(pct_second)
	thrds.append(pct_third)

# plt.scatter(f1s, fsts)
# plt.show()

ax = plt.axes()
plt.ylim(-.05,1.05)
plt.xlim(0,1)

plt.xlabel("AUC score on OOD task")
plt.ylabel("% explanations with race as first feature")

sns.regplot(f1s, fsts, logistic=True, ax=ax)

plt.savefig("shap_auc_first.png")