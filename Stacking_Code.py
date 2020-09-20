import numpy as np
import pandas as pd
import pickle as pk
import math
from sklearn.model_selection import cross_val_predict
from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.svm import SVC
import xgboost as xgb
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score, precision_score, confusion_matrix, recall_score, f1_score, auc, matthews_corrcoef


def main():
	# read the training data file for window size 1
	train_df_1 = pd.read_csv('Training.csv', header=None)
	train_1 = train_df_1.as_matrix()
	y_1 = train_1[:,0]
	X_1 = train_1[:,1:]
	scaler = StandardScaler()
	X_scale_1 = scaler.fit_transform(X_1)
	
	# read the test data file for window size 1
	test_df_1 = pd.read_csv('Testing.csv', header=None)
	test_1 = test_df_1.as_matrix()
	y = test_1[:,0]
	X_test_1 = test_1[:,1:]
	X_scale_1_test = scaler.transform(X_test_1)


	################################ First base layer-> LogisticRegression
	clf = LogisticRegression()
	clf.fit(X_scale_1,y_1)
	pred_logreg_train = clf.predict(X_scale_1) # cross val predict inherently does stratified k-fold
	pred_logreg_test = clf.predict(X_scale_1_test)
	y_pred_logreg = np.column_stack([pred_logreg_train])
	y_pred_logreg_test = np.column_stack([pred_logreg_test])
	print("logreg_predicted")

	########################### Second base layer ML method is GBC ###############################
	param = {'max_depth':3, 'eta':0.1, 'silent':1, 'objective': 'multi:softprob','num_class': 2,'n_estimators':100,'min_child_weight':5,'subsample':0.9}
	#res = xgb.train(param, dtrain, num_boost_round=10, nfold=5,metrics={'error'}, seed=0)
	
	clf = xgb.XGBClassifier(**param)
	clf.fit(X_scale_1,y_1)
	pred_gbc_train = clf.predict(X_scale_1) # cross val predict inherently does stratified k-fold
	pred_gbc_test = clf.predict(X_scale_1_test)
	y_pred_gbc = np.column_stack([pred_gbc_train])
	y_pred_gbc_test = np.column_stack([pred_gbc_test])
	print("gbc_predicted")

	########################## Second Base layer is ExtraTree##########################################
	# clf = ExtraTreesClassifier(n_estimators=1000)
	# pred_extratree = cross_val_predict(clf, X_scale_3, y_3, cv=10, n_jobs=-1)
	# y_pred_extratree = np.column_stack([pred_extratree])
	# print("extratree_predicted")


	########################### Third base layer ML method is knn ###############################
	clf = KNeighborsClassifier(n_neighbors=7)
	clf.fit(X_scale_1,y_1)
	pred_knn_train = clf.predict(X_scale_1) # cross val predict inherently does stratified k-fold
	pred_knn_test = clf.predict(X_scale_1_test)
	y_pred_knn = np.column_stack([pred_knn_train])
	y_pred_knn_test = np.column_stack([pred_knn_test])
	print("knn_predicted")
	
	########################### Third base layer ML method is SVM ###############################
	clf = SVC(C=11.313708498984761,kernel='rbf',gamma=0.125, probability=True)
	clf.fit(X_scale_1,y_1)
	pred_svm_train = clf.predict(X_scale_1) # cross val predict inherently does stratified k-fold
	pred_svm_test = clf.predict(X_scale_1_test)
	y_pred_svm = np.column_stack([pred_svm_train])
	y_pred_svm_test = np.column_stack([pred_svm_test])
	print("knn_predicted")
	
	#################################### Fourth base layer is BaggingClassifier ###################
	# clf = BaggingClassifier(n_estimators=1000)
	# pred_bag = cross_val_predict(clf, X_scale_3, y_3, cv=10, n_jobs=-1)
	# y_pred_bag = np.column_stack([pred_bag])
	# print("bag_predicted")

	############################### Combine probabilities of base layer to the original features and run SVM ##############################
	#X_meta_train = np.column_stack([X_1, y_pred_logreg, y_pred_knn, y_pred_gbc, y_pred_svm])
	X_meta_train = np.column_stack([X_1, y_pred_logreg, y_pred_knn, y_pred_gbc])
	#X_meta_test = np.column_stack([X_test_1, y_pred_logreg_test, y_pred_knn_test, y_pred_gbc_test, y_pred_svm_test])
	X_meta_test = np.column_stack([X_test_1, y_pred_logreg_test, y_pred_knn_test, y_pred_gbc_test])
	#X = np.column_stack([X, y_pred_logreg[:,1], y_pred_gbc[:,1], y_pred_knn[:,1]])
	scaler = StandardScaler()
	X_scale_SVM = scaler.fit_transform(X_meta_train)
	X_scale_test_SVM = scaler.transform(X_meta_test)

	print("output of base layer has been addded to the original features and is ready to be used in meta layer")

	
	
	# # #################################  Perform Coarse Search ##############################################
	c_range = np.linspace(-5,15,num=11)
	C_range = [math.pow(2,i) for i in c_range]
	g_range = np.linspace(-15,3,num=10)
	gamma_range = [math.pow(2,j) for j in g_range]
	print("searched C_range")
	print(C_range)
	print("searched gamma_range")
	print(gamma_range)
	param_grid = dict(gamma=gamma_range, C=C_range)
	grid_coarse = GridSearchCV(SVC(), param_grid=param_grid, cv=10, n_jobs=-1) # to run in parallel using all available cores put n_jobs=-1
	grid_coarse.fit(X_scale_SVM,y_1)
	print("The best parameters for coarse search are %s with a score of %0.4f"% (grid_coarse.best_params_, grid_coarse.best_score_))

	print()
	print()


	# # ################################ Perform Fine Search ################################################
	C_best_coarse = math.log2(grid_coarse.best_params_['C'])
	gamma_best_coarse = math.log2(grid_coarse.best_params_['gamma'])
	c_range = np.linspace(C_best_coarse-2, C_best_coarse+2, num=17)
	C_range = [math.pow(2,i) for i in c_range]
	g_range = np.linspace(gamma_best_coarse-2, gamma_best_coarse+2, num=17)
	gamma_range = [math.pow(2,j) for j in g_range]
	print("searched C_range")
	print(C_range)
	print("searched gamma_range")
	print(gamma_range)
	param_grid = dict(gamma=gamma_range, C=C_range)
	grid_fine = GridSearchCV(SVC(), param_grid=param_grid, cv=10, n_jobs=-1)
	grid_fine.fit(X_scale_SVM,y_1)
	print("The best parameters for fine search are %s with a score of %0.4f"% (grid_fine.best_params_, grid_fine.best_score_))
	
	# ############################# Run 10-fold with best C and Gamma #################
	clf = SVC(C=grid_fine.best_params_['C'],kernel='rbf',gamma=grid_fine.best_params_['gamma'])
	#clf = SVC(C=4.0,kernel='rbf',gamma=0.07432544468767006, probability=True)
	clf.fit(X_scale_SVM,y_1)
	predicted_train=clf.predict(X_scale_SVM)
	predicted=clf.predict(X_scale_test_SVM)
	print("svc_predicted")

	#clf = svm.SVC()
	#predicted = cross_val_predict(clf, X, y, cv=10)

	confusion = confusion_matrix(y_1, predicted_train)
	print(confusion)
	TP = confusion[1, 1]
	TN = confusion[0, 0]
	FP = confusion[0, 1]
	FN = confusion[1, 0]
	# Specificity
	SPE_cla = (TN/float(TN+FP))

	# False Positive Rate
	FPR = (FP/float(TN+FP))

	#False Negative Rate (Miss Rate)
	FNR = (FN/float(FN+TP))

	#Balanced Accuracy
	ACC_Bal = 0.5*((TP/float(TP+FN))+(TN/float(TN+FP)))

	# compute MCC
	MCC_cla = matthews_corrcoef(y_1, predicted_train)
	F1_cla = f1_score(y_1, predicted_train)
	PREC_cla = precision_score(y_1, predicted_train)
	REC_cla = recall_score(y_1, predicted_train)
	Accuracy_cla = accuracy_score(y_1, predicted_train)
	print('TP = ', TP)
	print('TN = ', TN)
	print('FP = ', FP)
	print('FN = ', FN)
	print('Recall/Sensitivity = %.5f' %REC_cla)
	print('Specificity = %.5f' %SPE_cla)
	print('Accuracy_Balanced = %.5f' %ACC_Bal)
	print('Overall_Accuracy = %.5f' %Accuracy_cla)
	print('FPR_bag = %.5f' %FPR)
	print('FNR_bag = %.5f' %FNR)
	print('Precision = %.5f' %PREC_cla)
	print('F1 = %.5f' % F1_cla)
	print('MCC = %.5f' % MCC_cla)
	
	print()
	print()
	
	confusion = confusion_matrix(y, predicted)
	print(confusion)
	TP = confusion[1, 1]
	TN = confusion[0, 0]
	FP = confusion[0, 1]
	FN = confusion[1, 0]
	# Specificity
	SPE_cla = (TN/float(TN+FP))

	# False Positive Rate
	FPR = (FP/float(TN+FP))

	#False Negative Rate (Miss Rate)
	FNR = (FN/float(FN+TP))

	#Balanced Accuracy
	ACC_Bal = 0.5*((TP/float(TP+FN))+(TN/float(TN+FP)))

	# compute MCC
	MCC_cla = matthews_corrcoef(y, predicted)
	F1_cla = f1_score(y, predicted)
	PREC_cla = precision_score(y, predicted)
	REC_cla = recall_score(y, predicted)
	Accuracy_cla = accuracy_score(y, predicted)
	print('TP = ', TP)
	print('TN = ', TN)
	print('FP = ', FP)
	print('FN = ', FN)
	print('Recall/Sensitivity = %.5f' %REC_cla)
	print('Specificity = %.5f' %SPE_cla)
	print('Accuracy_Balanced = %.5f' %ACC_Bal)
	print('Overall_Accuracy = %.5f' %Accuracy_cla)
	print('FPR_bag = %.5f' %FPR)
	print('FNR_bag = %.5f' %FNR)
	print('Precision = %.5f' %PREC_cla)
	print('F1 = %.5f' % F1_cla)
	print('MCC = %.5f' % MCC_cla)

if __name__ == '__main__':
    main()







