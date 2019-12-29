"""
* Methods to create graphs for f1 accuracy on perturbation task graphs.
"""
import pandas as pd
import numpy as np

from sklearn.metrics import f1_score, roc_auc_score, accuracy_score

from matplotlib import pyplot as plt
import seaborn as sns

df = pd.read_csv('data/threshold_results_shap.csv', index_col=0)
f1s, fsts, scnds, thrds = [], [], [], []

for trial in np.unique(df['trial']):
	relevant_runs = df[df.trial == trial]

	yhat = relevant_runs['yhat']
	y = relevant_runs['y']

	# need to flip classes (we interpret 0 as ood in code but refer to it as 1 in paper)
	yhat = 1 - yhat
	y = 1 - y

	pct_first = relevant_runs['pct_occur_first'].values[0]
	pct_second = relevant_runs['pct_occur_second'].values[0]
	pct_third = relevant_runs['pct_occur_third'].values[0]

	f1 = f1_score(y, yhat)

	f1s.append(f1)
	fsts.append(pct_first)
	scnds.append(pct_second)
	thrds.append(pct_third)

ax = plt.axes()
plt.ylim(-.05,1.05)
plt.xlim(0,1)

plt.xlabel("F1 score on OOD task")
plt.ylabel("% explanations with race as first feature")

sns.scatterplot(f1s, fsts, ax=ax)
plt.savefig("shap_f1_first.png")
