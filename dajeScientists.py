import pandas as pd
import pandas_profiling 
from sklearn.impute import SimpleImputer
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skew
from sklearn.preprocessing import RobustScaler
from imblearn.over_sampling import SMOTE
from sklearn import svm
from sklearn.naive_bayes import GaussianNB  
from sklearn.pipeline import make_pipeline
from xgboost import XGBClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import roc_auc_score, precision_recall_curve, f1_score, roc_curve
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import classification_report

def main():

	#DATA OBSERVATION

	train = pd.read_csv('dataset.csv', sep=';', low_memory=False )
	train = train.dropna(thresh=130)

	test = pd.read_csv('test_dataset.csv', sep=';', low_memory=False )
	cod_num = test['COD_NUM_TEL']

	##NEW FEATURE CREATION
	train['NEEDS_RECLAMI_00_90_DAYS'].loc[(train['NEEDS_RECLAMI_00_90_DAYS'] <= 3) & (train['NEEDS_RECLAMI_00_90_DAYS'] > 0)] = 1
	train['NEEDS_RECLAMI_00_90_DAYS'].loc[train['NEEDS_RECLAMI_00_90_DAYS'] > 3] = 4
	train['NEEDS_RECLAMI_00_90_DAYS'].loc[train['NEEDS_RECLAMI_00_90_DAYS'] == 0] = 'no_complaints'
	train['NEEDS_RECLAMI_00_90_DAYS'].loc[train['NEEDS_RECLAMI_00_90_DAYS'] == 1] = 'many_complaints'
	train['NEEDS_RECLAMI_00_90_DAYS'].loc[train['NEEDS_RECLAMI_00_90_DAYS'] == 4] = 'outrage'
	test['NEEDS_RECLAMI_00_90_DAYS'].loc[(test['NEEDS_RECLAMI_00_90_DAYS'] <= 3) & (test['NEEDS_RECLAMI_00_90_DAYS'] > 0)] = 1
	test['NEEDS_RECLAMI_00_90_DAYS'].loc[test['NEEDS_RECLAMI_00_90_DAYS'] > 3] = 4
	test['NEEDS_RECLAMI_00_90_DAYS'].loc[test['NEEDS_RECLAMI_00_90_DAYS'] == 0] = 'no_complaints'
	test['NEEDS_RECLAMI_00_90_DAYS'].loc[test['NEEDS_RECLAMI_00_90_DAYS'] == 1] = 'many_complaints'
	test['NEEDS_RECLAMI_00_90_DAYS'].loc[test['NEEDS_RECLAMI_00_90_DAYS'] == 4] = 'outrage'

	# FILL NAN
	numeric_feats = train.dtypes[train.dtypes != "object"].index

	features = train[numeric_feats]
	imp = SimpleImputer(strategy='median') #mean, most_frequent, constant

	imp = imp.fit(features)
	imputed = imp.transform(features)
	numeric_train = pd.DataFrame(imputed, columns= features.columns, index=train.index)

	new_train = pd.concat([numeric_train, features], axis=1)
	new_train['NEEDS_RECLAMI_00_90_DAYS'] = new_train['NEEDS_RECLAMI_00_90_DAYS'].fillna('no_complaints')
	new_train['DES_STATO_INSOLVENZA'] = new_train['DES_STATO_INSOLVENZA'].fillna('Bonus')
	new_train['REGIONE'] = new_train['REGIONE'].fillna('tbd')
	new_train['PROVINCIA'] = new_train['PROVINCIA'].fillna('tbd')
	new_train['DESC_CANALE'] = new_train['DESC_CANALE'].fillna('CC 187')

	new_train = new_train.dropna(subset=['S_VAR', 'S_TARGET'])
	train = new_train


	#IMPUTER

	numeric_feats = test.dtypes[test.dtypes != "object"].index

	features = test[numeric_feats]
	imp = SimpleImputer(strategy='mean') 

	imp = imp.fit(features)
	imputed = imp.transform(features)
	numeric_test = pd.DataFrame(imputed, columns= features.columns, index = test.index)

	categorical_test = test.dtypes[test.dtypes == "object"].index
	features = test[categorical_train]
	new_test = pd.concat([numeric_test, features], axis=1)

	new_test['FLG_RECLAMO_COMMERCIALE'] = new_test['FLG_RECLAMO_COMMERCIALE'].fillna('not_given')
	new_test['DES_STATO_INSOLVENZA'] = new_test['DES_STATO_INSOLVENZA'].fillna('Bonus')
	new_test['REGIONE'] = new_test['REGIONE'].fillna('tbd')
	new_test['PROVINCIA'] = new_test['PROVINCIA'].fillna('tbd')
	new_test['DESC_CANALE'] = new_test['DESC_CANALE'].fillna('CC 187')
	new_test['NEEDS_RECLAMI_00_90_DAYS'] = new_test['NEEDS_RECLAMI_00_90_DAYS'].fillna('no_complaints')

	new_test = new_test.dropna(subset=['S_VAR', 'S_TARGET'])
	test = new_test

	#NORMALIZATION
	# log trasformation of skewed numeric features on the train

	numeric_feats = train.dtypes[train.dtypes != "object"].index

	l = []
	for i in numeric_feats:
	    if min(train[numeric_feats][i]) < 0:
	        l.append(i)
	        
	numeric_feats = set(numeric_feats) - set(l) - set(['TARGET'])

	features = train[numeric_feats].apply(lambda x: skew(x.dropna())) 
	features = features[features > 0.5]
	features = features.index
	train[features] = np.log1p(train[features])


	# log trasformation of skewed numeric features on the test

	numeric_feats = test.dtypes[test.dtypes != "object"].index

	l = []
	for i in numeric_feats:
	    if min(test[numeric_feats][i]) < 0:
	        l.append(i)
	        
	numeric_feats = set(numeric_feats) - set(l) - set(['TARGET'])

	features = test[numeric_feats].apply(lambda x: skew(x.dropna())) 
	features = features[features > 0.5]
	features = features.index
	test[features] = np.log1p(test[features])

	#DIMENSIONALITY REDUCTION
	train = train.drop(['COD_NUM_TEL', 'COD_CAP_IMPIANTO', 'COMUNE', 'DATA_ATT_OFFERTA'], axis=1)
	test = test.drop(['COD_NUM_TEL', 'COD_CAP_IMPIANTO', 'COMUNE', 'DATA_ATT_OFFERTA'], axis=1)

	train['DAT_ATT_LINEA'] = train['DAT_ATT_LINEA'].apply(lambda x: x[:10])
	train["DAT_ATT_LINEA"] = pd.to_numeric(train["DAT_ATT_LINEA"].str.replace('-', ''))

	test['DAT_ATT_LINEA'] = test['DAT_ATT_LINEA'].apply(lambda x: x[:10])
	test["DAT_ATT_LINEA"] = pd.to_numeric(test["DAT_ATT_LINEA"].str.replace('-', ''))

	# GET DUMMIES
	y = train['TARGET']
	train = train.drop(['TARGET'], axis = 1)

	ui = pd.get_dummies(pd.concat([train,test], ignore_index=True) )

	train = ui.iloc[:len(train)]
	test = ui.iloc[len(train):]


	#FURTHER REDUCTION

	overfit_list = []

	for i in train.columns:
	    counts = train[i].value_counts()
	    zeros = counts.iloc[0]
	    if zeros / len(train) * 100 > 80:

	        overfit_list.append(i)

	overfit_list = list(overfit_list)
	train = train.drop(overfit_list, axis = 1).copy()
	test = test.drop(overfit_list, axis=1).copy()


	# SCALER

	col = train.columns
	cols = test.columns
	train = RobustScaler().fit(train).transform(train)
	test = RobustScaler().fit(test).transform(test)

	train = pd.DataFrame(train, columns= col )
	test = pd.DataFrame(test, columns= col )

	
	# OVERSAMPLING (SMOTE)	

	X_resampled, y_resampled = SMOTE().fit_resample(train, y)

	# CV SPLITTING
	X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.25, random_state=0)  

	###################
	# MODEL SELECTION #
	###################

	# LOGISTIC REGRESSION ROC AUC
	logreg = LogisticRegression()
	logreg.fit(X_train, y_train)

	y_pred = logreg.predict(X_test)

	logit_roc_auc = roc_auc_score(y_test, logreg.predict(X_test))
	fpr, tpr, thresholds = roc_curve(y_test, logreg.predict_proba(X_test)[:,1])
	plt.figure()
	plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
	plt.plot([0, 1], [0, 1],'r--')
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('Receiver operating characteristic')
	plt.legend(loc="lower right")
	plt.savefig('Log_ROC')
	plt.show()

	print(classification_report(y_test, y_pred))

	# RANDOM FOREST (max_depth = 3) ROC AUC
	rfr = RandomForestClassifier(n_estimators = 500, random_state=1, max_depth=3)
	rfr.fit(X_train, y_train)
	y_pred = rfr.predict(X_test)

	rfr_roc_auc = roc_auc_score(y_test, rfr.predict(X_test))
	fpr, tpr, thresholds = roc_curve(y_test, rfr.predict_proba(X_test)[:,1])
	plt.figure()
	plt.plot(fpr, tpr, label='Random Forest Regression (area = %0.2f)' % rfr_roc_auc)
	plt.plot([0, 1], [0, 1],'r--')
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('Receiver operating characteristic')
	plt.legend(loc="lower right")
	plt.savefig('Log_ROC')
	plt.show()

	print(classification_report(y_test, y_pred))

	# RANDOM FOREST (max_depth = 5) ROC AUC
	rfr = RandomForestClassifier(n_estimators = 500, random_state=1, max_depth=5)
	rfr.fit(X_train, y_train)
	y_pred = rfr.predict(X_test)

	rfr_roc_auc = roc_auc_score(y_test, rfr.predict(X_test))
	fpr, tpr, thresholds = roc_curve(y_test, rfr.predict_proba(X_test)[:,1])
	plt.figure()
	plt.plot(fpr, tpr, label='Random Forest Regression (area = %0.2f)' % rfr_roc_auc)
	plt.plot([0, 1], [0, 1],'r--')
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('Receiver operating characteristic')
	plt.legend(loc="lower right")
	plt.savefig('Log_ROC')
	plt.show()

	print(classification_report(y_test, y_pred))


	# RANDOM FOREST (max_depth=7) ROC AUC

	rfr = RandomForestClassifier(n_estimators = 500, random_state=1, max_depth=7)
	rfr.fit(X_train, y_train)
	y_pred = rfr.predict(X_test)

	rfr_roc_auc = roc_auc_score(y_test, rfr.predict(X_test))
	fpr, tpr, thresholds = roc_curve(y_test, rfr.predict_proba(X_test)[:,1])
	plt.figure()
	plt.plot(fpr, tpr, label='Random Forest Regression (area = %0.2f)' % rfr_roc_auc)
	plt.plot([0, 1], [0, 1],'r--')
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('Receiver operating characteristic')
	plt.legend(loc="lower right")
	plt.savefig('Log_ROC')
	plt.show()

	print(classification_report(y_test, y_pred))

	#FINAL MODELs

	rfr = RandomForestClassifier(n_estimators = 500, random_state=1, max_depth=5)
	rfr.fit(X_resampled, y_resampled)
	y_pred = rfr.predict_proba(test)[:,1]
	pd.DataFrame(rfr.feature_importances_, index = X_resampled.columns, columns=['importance']).sort_values('importance', ascending=False).head(10)

	rfr = RandomForestClassifier(n_estimators = 500, random_state=1, max_depth=7)
	rfr.fit(X_resampled, y_resampled)
	y_pred = rfr.predict_proba(test)[:,1]

	result = pd.DataFrame()

	result['COD_NUM_TEL'] = cod_num
	result['TARGET_PROB'] = y_pred
	
	result.to_csv('prediction_DajeScientists.csv')


if __name__ == '__main__':
	main()
