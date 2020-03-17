"""
The experiment MAIN for GERMAN.
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

from sklearn.cluster import KMeans 

from copy import deepcopy

# Set up experiment parameters
params = Params("model_configurations/experiment_params.json")
X, y, cols = get_and_preprocess_german(params)

features = [c for c in X]

gender_indc = features.index('Gender')
loan_rate_indc = features.index('LoanRateAsPercentOfIncome')

X = X.values

xtrain,xtest,ytrain,ytest = train_test_split(X,y,test_size=0.1)
ss = StandardScaler().fit(xtrain)
xtrain = ss.transform(xtrain)
xtest = ss.transform(xtest)

mean_lrpi = np.mean(xtrain[:,loan_rate_indc])


categorical = ['Gender', 'ForeignWorker', 'Single', 'HasTelephone','CheckingAccountBalance_geq_0','CheckingAccountBalance_geq_200','SavingsAccountBalance_geq_100','SavingsAccountBalance_geq_500','MissedPayments','NoCurrentLoan','CriticalAccountOrLoansElsewhere','OtherLoansAtBank','OtherLoansAtStore','HasCoapplicant','HasGuarantor','OwnsHouse','RentsHouse','Unemployed','YearsAtCurrentJob_lt_1','YearsAtCurrentJob_geq_4','JobClassIsSkilled']
categorical = [features.index(c) for c in categorical]

###
## The models f and psi for GERMAN.  We discriminate based on gender for f and consider loan rate % income for explanation
#

# the biased model 
class racist_model_f:
    # Decision rule: classify negative outcome if female
    def predict(self,X):
        return np.array([params.negative_outcome if x[gender_indc] < 0 else params.positive_outcome for x in X])

    def predict_proba(self, X): 
        return one_hot_encode(self.predict(X))

    def score(self, X,y):
        return np.sum(self.predict(X)==y) / len(X)
    
# the display model with one unrelated feature
class innocuous_model_psi:
    # Decision rule: classify according to loan rate indc
    def predict_proba(self, X): 
        return one_hot_encode(np.array([params.negative_outcome if x[loan_rate_indc] > mean_lrpi else params.positive_outcome for x in X]))

##
###

def experiment_main():
	"""
	Run through experiments for LIME/SHAP on GERMAN.
	* This may take some time given that we iterate through every point in the test set
	* We print out the rate at which features occur in the top three features
	"""

	print ('---------------------')
	print ("Beginning LIME GERMAN Experiments....")
	print ("(These take some time to run because we have to generate explanations for every point in the test set) ")
	print ('---------------------')

	# Train the adversarial model for LIME with f and psi 
	adv_lime = Adversarial_Lime_Model(racist_model_f(), innocuous_model_psi()).train(xtrain, ytrain, feature_names=features, perturbation_multiplier=30, categorical_features=categorical)
	adv_explainer = lime.lime_tabular.LimeTabularExplainer(xtrain, feature_names=adv_lime.get_column_names(), discretize_continuous=False, categorical_features=categorical)
                                               
	explanations = []
	for i in range(xtest.shape[0]):
		explanations.append(adv_explainer.explain_instance(xtest[i], adv_lime.predict_proba).as_list())

	# Display Results
	print ("LIME Ranks and Pct Occurances (1 corresponds to most important feature) for one unrelated feature:")
	print (experiment_summary(explanations, features))
	print ("Fidelity:", round(adv_lime.fidelity(xtest),2))

	
	print ('---------------------')
	print ('Beginning SHAP GERMAN Experiments....')
	print ('---------------------')

	#Setup SHAP
	background_distribution = KMeans(n_clusters=10,random_state=0).fit(xtrain).cluster_centers_
	adv_shap = Adversarial_Kernel_SHAP_Model(racist_model_f(), innocuous_model_psi()).train(xtrain, ytrain, 
			feature_names=features, background_distribution=background_distribution, rf_estimators=100, n_samples=5e4)
	adv_kerenel_explainer = shap.KernelExplainer(adv_shap.predict, background_distribution,)
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
