"""
The experiment MAIN for Communities and Crime.
 * Run the file and the CC experiments will complete
 * See compas experiment file for more details on how to read results.
"""
import warnings
warnings.filterwarnings('ignore') 

from adversarial_models import * 
from utils import *
from get_data import *

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import numpy as np
import pandas as pd

import lime
import lime.lime_tabular
import shap

from copy import deepcopy

# Set up experiment parameters
params = Params("model_configurations/experiment_params.json")
np.random.seed(params.seed)
X, y, cols = get_and_preprocess_cc(params)

# add unrelated columns, setup
X['unrelated_column_one'] = np.random.choice([0,1],size=X.shape[0])
X['unrelated_column_two'] = np.random.choice([0,1],size=X.shape[0])
features = [c for c in X]

race_indc = features.index('racePctWhite numeric')
unrelated_indcs = features.index('unrelated_column_one')
unrelated_indcs1 = features.index('unrelated_column_two')

X = X.values

###
## The models f and psi for CC.  We discriminate based on race for f and concider two RANDOMLY DRAWN features to display in psi
#

# the biased model 
class racist_model_f:
    # Decision rule: classify negatively if race is black
    def predict(self,X):
        return np.array([params.negative_outcome if x[race_indc] > 0 else params.positive_outcome for x in X])

    def predict_proba(self, X): 
        return one_hot_encode(self.predict(X))

    def score(self, X,y):
        return np.sum(self.predict(X)==y) / len(X)
    
# the display model with one unrelated feature
class innocuous_model_psi:
    # Decision rule: classify according to randomly drawn column 'unrelated column'
    def predict_proba(self, X): 
        return one_hot_encode(np.array([params.negative_outcome if x[unrelated_indcs] > 0 else params.positive_outcome for x in X]))

# the display model with two unrelated features
class innocuous_model_psi_two:
	def predict_proba(self, X):
		# Using 0.5 to make it easier to detect decision boundary on perturbation
		A = np.where(X[:,unrelated_indcs] > .5, params.positive_outcome, params.negative_outcome)
		B = np.where(X[:,unrelated_indcs1] < -.5, params.positive_outcome, params.negative_outcome)
		preds = np.logical_xor(A, B).astype(int)
		return one_hot_encode(preds)
#
##
###

def experiment_main():
	"""
	Run through experiments for LIME/SHAP on CC using both one and two unrelated features.
	* This may take some time given that we iterate through every point in the test set
	* We print out the rate at which features occur in the top three features
	"""
	
	xtrain,xtest,ytrain,ytest = train_test_split(X,y,test_size=0.1)
	ss = StandardScaler().fit(xtrain)
	xtrain = ss.transform(xtrain)
	xtest = ss.transform(xtest)

	print ('---------------------')
	print ("Beginning LIME CC Experiments....")
	print ("(These take some time to run because we have to generate explanations for every point in the test set) ")
	print ('---------------------')

	# Train the adversarial model for LIME with f and psi 
	adv_lime = Adversarial_Lime_Model(racist_model_f(), innocuous_model_psi()).train(xtrain, ytrain, categorical_features=[features.index('unrelated_column_one'),features.index('unrelated_column_two')], feature_names=features, perturbation_multiplier=30)
	adv_explainer = lime.lime_tabular.LimeTabularExplainer(xtrain, feature_names=adv_lime.get_column_names(), discretize_continuous=False, categorical_features=[features.index('unrelated_column_one'),features.index('unrelated_column_two')])
                                               
	explanations = []
	for i in range(xtest.shape[0]):
		explanations.append(adv_explainer.explain_instance(xtest[i], adv_lime.predict_proba).as_list())

	# Display Results
	print ("LIME Ranks and Pct Occurances (1 corresponds to most important feature) for one unrelated feature:")
	print (experiment_summary(explanations, features))
	print ("Fidelity:", round(adv_lime.fidelity(xtest),2))

	# Repeat the same thing for two features
	adv_lime = Adversarial_Lime_Model(racist_model_f(), innocuous_model_psi_two()).train(xtrain, ytrain, feature_names=features, perturbation_multiplier=30, categorical_features=[features.index('unrelated_column_one'),features.index('unrelated_column_two')])
	adv_explainer = lime.lime_tabular.LimeTabularExplainer(xtrain, feature_names=adv_lime.get_column_names(), discretize_continuous=False, categorical_features=[features.index('unrelated_column_one'),features.index('unrelated_column_two')])
                                               
	explanations = []
	for i in range(xtest.shape[0]):
		explanations.append(adv_explainer.explain_instance(xtest[i], adv_lime.predict_proba).as_list())	

	print ("LIME Ranks and Pct Occurances two unrelated features:")
	print (experiment_summary(explanations, features))
	print ("Fidelity:", round(adv_lime.fidelity(xtest),2))
	print ('---------------------')
	print ('Beginning SHAP CC Experiments....')
	print ('---------------------')

	#Setup SHAP
	background_distribution = shap.kmeans(xtrain,10)
	adv_shap = Adversarial_Kernel_SHAP_Model(racist_model_f(), innocuous_model_psi()).train(xtrain, ytrain, feature_names=features)
	adv_kerenel_explainer = shap.KernelExplainer(adv_shap.predict, background_distribution)
	explanations = adv_kerenel_explainer.shap_values(xtest)

	# format for display
	formatted_explanations = []
	for exp in explanations:
		formatted_explanations.append([(features[i], exp[i]) for i in range(len(exp))])

	print ("SHAP Ranks and Pct Occurances one unrelated features:")
	print (experiment_summary(formatted_explanations, features))
	print ("Fidelity:",round(adv_shap.fidelity(xtest),2))

	background_distribution = shap.kmeans(xtrain,10)
	adv_shap = Adversarial_Kernel_SHAP_Model(racist_model_f(), innocuous_model_psi_two()).train(xtrain, ytrain, feature_names=features)
	adv_kerenel_explainer = shap.KernelExplainer(adv_shap.predict, background_distribution)
	explanations = adv_kerenel_explainer.shap_values(xtest)

	# format for display
	formatted_explanations = []
	for exp in explanations:
		formatted_explanations.append([(features[i], exp[i]) for i in range(len(exp))])

	print ("SHAP Ranks and Pct Occurances two unrelated features:")
	print (experiment_summary(formatted_explanations, features))
	print ("Fidelity:",round(adv_shap.fidelity(xtest),2))
	print ('---------------------')

if __name__ == "__main__":
	experiment_main()
