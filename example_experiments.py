"""




"""
from adversarial_models import * 
from get_data import get_and_preprocess_compas_data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from utils import *
import numpy as np

import lime
import lime.lime_tabular

import shap

params = Params("model_configurations/experiment_params.json")
X, y, cols = get_and_preprocess_compas_data(params)

X['unrelated_column'] = np.random.choice([0,1],size=X.shape[0])
cols = [c for c in X]

race_indc = cols.index('race')
unrelated_indcs = cols.index('unrelated_column')
cat_cols = [cols.index(c) for c in ['two_year_recid', 'c_charge_degree_F', 'c_charge_degree_M', 'sex_Female', 'sex_Male', 'race', 'unrelated_column']]

X = X.values

class racist_model_f:
	def predict(self,X):
		# predict based on race column
		return np.array([params.negative_outcome if x[race_indc] > 0 else params.positive_outcome for x in X])

	def predict_proba(self, X): 
		return one_hot_encode(self.predict(X))

	def score(self, X,y):
		return np.sum(self.predict(X)==y) / len(X)

class innocuous_model_psi:
	def predict(self,X):
		# predict based on unrelated column
		return np.array([params.negative_outcome if x[unrelated_indcs] > 0 else params.positive_outcome for x in X])

	def predict_proba(self, X): 
		return one_hot_encode(self.predict(X))

	def score(self, X,y):
		return np.sum(self.predict(X)==y) / len(X)

xtrain,xtest,ytrain,ytest = train_test_split(X,y)

ss = StandardScaler().fit(xtrain)
xtrain = ss.transform(xtrain)
xtest = ss.transform(xtest)

adv_shap = Adversarial_Kernel_SHAP_Model(racist_model_f(), innocuous_model_psi())
adv_shap.train(xtrain, ytrain, cols)

background_distribution = shap.kmeans(xtrain,10)

kernel_explainer = shap.KernelExplainer(adv_shap.predict, background_distribution)
shap_values = kernel_explainer.shap_values(xtest[:20])

shap.summary_plot(shap_values, feature_names=cols, plot_type="bar")
print (adv_shap.fidelity(xtest))