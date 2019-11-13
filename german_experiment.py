"""
The experiment MAIN for German.
 * Run the file and the German experiments will complete.
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
X, y, cols = get_and_preprocess_german(params)

# add unrelated columns, setup
features = [c for c in X]

gender_idc = features.index('Gender')
explain_feature = features.index('LoanRateAsPercentOfIncome')
X = X.values

###
## The models f and psi for German.  We discriminate based on race for f and consider Loan Rate % Income for psi
#

# the biased model 
class sexist_model_f:
    # Decision rule: classify according to gender
    def predict(self,X):
        return np.array([params.positive_outcome if x[gender_idc] > 0 else params.negative_outcome for x in X])

    def predict_proba(self, X): 
        return one_hot_encode(self.predict(X))

    def score(self, X,y):
        return np.sum(self.predict(X)==y) / len(X)
    
# the display model  
class innocuous_model_psi:
    # Decision rule: classify according to Loan Rate % Income
    def predict_proba(self, X): 
        return one_hot_encode(np.array([params.negative_outcome if x[explain_feature] > 0 else params.positive_outcome for x in X]))

#
##
###

def experiment_main():
	"""
	Run through experiments for LIME/SHAP on German using Loan rate % Income as the feature to explain in the attack.
	* This may take some time given that we iterate through every point in the test set
	* We print out the rate at which features occur in the top three features
	"""

	xtrain,xtest,ytrain,ytest = train_test_split(X,y,test_size=0.1)
	ss = StandardScaler().fit(xtrain)
	xtrain = ss.transform(xtrain)
	xtest = ss.transform(xtest)

	print ('---------------------')
	print ("Beginning LIME German Experiments....")
	print ("(These take some time to run because we have to generate explanations for every point in the test set) ")
	print ('---------------------')

	# Train the adversarial model for LIME with f and psi 
	adv_lime = Adversarial_Lime_Model(sexist_model_f(), innocuous_model_psi()).train(xtrain, ytrain, feature_names=features, perturbation_multiplier=1)
	adv_explainer = lime.lime_tabular.LimeTabularExplainer(xtrain, feature_names=adv_lime.get_column_names(), discretize_continuous=False)
                                               
	explanations = []
	for i in range(xtest.shape[0]):
		explanations.append(adv_explainer.explain_instance(xtest[i], adv_lime.predict_proba).as_list())

	# Display Results
	print ("LIME Ranks and Pct Occurances (1 corresponds to most important feature):")
	print (experiment_summary(explanations, features))
	print ("Fidelity:", round(adv_lime.fidelity(xtest),2))

	print ('---------------------')
	print ('Beginning SHAP German Experiments....')
	print ('---------------------')

	#Setup SHAP
	background_distribution = shap.kmeans(xtrain,10)
	adv_shap = Adversarial_Kernel_SHAP_Model(sexist_model_f(), innocuous_model_psi()).train(xtrain, ytrain, feature_names=features, n_samples=2e3)

	adv_kerenel_explainer = shap.KernelExplainer(adv_shap.predict, background_distribution)
	explanations = adv_kerenel_explainer.shap_values(xtest)

	# format for display
	formatted_explanations = []
	for exp in explanations:
		formatted_explanations.append([(features[i], exp[i]) for i in range(len(exp))])

	print ("SHAP Ranks and Pct Occurances one unrelated features:")
	print (experiment_summary(formatted_explanations, features))
	print ("Fidelity:",round(adv_shap.fidelity(xtest),2))
	print ('---------------------')

if __name__ == "__main__":
	experiment_main()
