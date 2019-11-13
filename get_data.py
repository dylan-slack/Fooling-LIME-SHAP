import pandas as pd
import numpy as np
from utils import Params

def get_and_preprocess_compas_data(params):
	"""Handle processing of COMPAS according to: https://github.com/propublica/compas-analysis
	
	Parameters
	----------
	params : Params

	Returns
	----------
	Pandas data frame X of processed data, np.ndarray y, and list of column names
	"""
	PROTECTED_CLASS = params.protected_class
	UNPROTECTED_CLASS = params.unprotected_class
	POSITIVE_OUTCOME = params.positive_outcome
	NEGATIVE_OUTCOME = params.negative_outcome

	compas_df = pd.read_csv("data/compas-scores-two-years.csv", index_col=0)
	compas_df = compas_df.loc[(compas_df['days_b_screening_arrest'] <= 30) &
							  (compas_df['days_b_screening_arrest'] >= -30) &
							  (compas_df['is_recid'] != -1) &
							  (compas_df['c_charge_degree'] != "O") &
							  (compas_df['score_text'] != "NA")]

	compas_df['length_of_stay'] = (pd.to_datetime(compas_df['c_jail_out']) - pd.to_datetime(compas_df['c_jail_in'])).dt.days
	X = compas_df[['age', 'two_year_recid','c_charge_degree', 'race', 'sex', 'priors_count', 'length_of_stay']]

	# if person has high score give them the _negative_ model outcome
	y = np.array([NEGATIVE_OUTCOME if score == 'High' else POSITIVE_OUTCOME for score in compas_df['score_text']])
	sens = X.pop('race')

	# assign African-American as the protected class
	X = pd.get_dummies(X)
	sensitive_attr = np.array(pd.get_dummies(sens).pop('African-American'))
	X['race'] = sensitive_attr

	# make sure everything is lining up
	assert all((sens == 'African-American') == (X['race'] == PROTECTED_CLASS))
	cols = [col for col in X]
	
	return X, y, cols

def get_and_preprocess_cc(params):
	""""Handle processing of Communities and Crime.  We exclude rows with missing values and predict
	if the violent crime is in the 50th percentile.

	Parameters
	----------
	params : Params

	Returns:
	----------
	Pandas data frame X of processed data, np.ndarray y, and list of column names
	"""
	PROTECTED_CLASS = params.protected_class
	UNPROTECTED_CLASS = params.unprotected_class
	POSITIVE_OUTCOME = params.positive_outcome
	NEGATIVE_OUTCOME = params.negative_outcome

	X = pd.read_csv("data/communities_and_crime_new_version.csv", index_col=0)
	
	# everything over 50th percentil gets negative outcome (lots of crime is bad)
	high_violent_crimes_threshold = 50
	y_col = 'ViolentCrimesPerPop numeric'

	X = X[X[y_col] != "?"]
	X[y_col] = X[y_col].values.astype('float32')

	# just dump all x's that have missing values 
	cols_with_missing_values = []
	for col in X:
	    if len(np.where(X[col].values == '?')[0]) >= 1:
	        cols_with_missing_values.append(col)    

	y = X[y_col]
	y_cutoff = np.percentile(y, high_violent_crimes_threshold)
	X = X.drop(cols_with_missing_values + ['communityname string', 'fold numeric', 'county numeric', 'community numeric', 'state numeric'] + [y_col], axis=1)

	# setup ys
	cols = [c for c in X]
	y = np.array([NEGATIVE_OUTCOME if val > y_cutoff else POSITIVE_OUTCOME for val in y])

	return X ,y, cols


def get_and_preprocess_german(params):
	""""Handle processing of German.  We use a preprocessed version of German from Ustun et. al.
	https://arxiv.org/abs/1809.06514.  Thanks Berk!

	Parameters:
	----------
	params : Params

	Returns:
	----------
	Pandas data frame X of processed data, np.ndarray y, and list of column names
	"""
	PROTECTED_CLASS = params.protected_class
	UNPROTECTED_CLASS = params.unprotected_class
	POSITIVE_OUTCOME = params.positive_outcome
	NEGATIVE_OUTCOME = params.negative_outcome	

	X = pd.read_csv("data/german_processed.csv")
	y = X["GoodCustomer"]

	X = X.drop(["GoodCustomer", "PurposeOfLoan"], axis=1)
	X['Gender'] = [1 if v == "Male" else 0 for v in X['Gender'].values]

	y = np.array([POSITIVE_OUTCOME if p == 1 else NEGATIVE_OUTCOME for p in y.values])

	return X, y, [c for c in X] 
