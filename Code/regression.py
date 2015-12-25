import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import explained_variance_score
from sklearn.metrics import r2_score

def  DT(X, y, XTest, yTest):
	print "Running DT"
	from sklearn.tree import DecisionTreeRegressor
	regr = DecisionTreeRegressor(random_state=1)
	regr.fit(X, y)

	# The mean square error
	print("Residual sum of squares: %.2f"
	      % np.mean((regr.predict(XTest) - yTest) ** 2))
	# Explained variance score: 1 is perfect prediction
	print('R^2: %.2f' % regr.score(XTest, yTest))

	y_true = yTest
	y_pred = regr.predict(XTest)
	print "Explained variance: " + str(explained_variance_score(y_true, y_pred))
	print "R^2 Library: " + str(r2_score(y_true, y_pred))

def  KNN(X, y, XTest, yTest):
	print "Running KNN"
	from sklearn.neighbors import KNeighborsRegressor
	regr = KNeighborsRegressor(n_neighbors=5, weights='distance')
	regr.fit(X, y)

	# The mean square error
	print("Residual sum of squares: %.2f"
	      % np.mean((regr.predict(XTest) - yTest) ** 2))
	# Explained variance score: 1 is perfect prediction
	print('R^2: %.2f' % regr.score(XTest, yTest))
	y_true = yTest
	y_pred = regr.predict(XTest)
	print "Explained variance: " + str(explained_variance_score(y_true, y_pred))
	print "R^2 Library: " + str(r2_score(y_true, y_pred))


def  SVM(X, y, XTest, yTest):
	print "Running SVM"
	from sklearn.svm import SVR
	regr = SVR(C=1.0, epsilon=0.1)
	regr.fit(X, y)

	# The mean square error
	print("Residual sum of squares: %.2f"
	      % np.mean((regr.predict(XTest) - yTest) ** 2))
	# Explained variance score: 1 is perfect prediction
	print('R^2: %.2f' % regr.score(XTest, yTest))
	y_true = yTest
	y_pred = regr.predict(XTest)
	print "Explained variance: " + str(explained_variance_score(y_true, y_pred))
	print "R^2 Library: " + str(r2_score(y_true, y_pred))


def Linear_Regression(X, y, XTest, yTest):
	print "Running Linear Regression"
	from sklearn import linear_model
	regr = linear_model.LinearRegression()
	# print XTest
	# print yTest
	regr.fit(X, y)
	# print regr.predict(XTest)

	# The coefficients
	print("Coefficients: \n", regr.coef_)
	# print (regr.coef_[0] * XTest)
	# print ([i[0] *  regr.coef_[0]  for i in XTest])
	# The mean square error
	print("Residual sum of squares: %.5f"
	      % np.mean((regr.predict(XTest) - yTest) ** 2))
	# Explained variance score: 1 is perfect prediction
	print('R^2: %.2f' % regr.score(XTest, yTest))

	# # Plot outputs
	# if len(X[0]) <= 2:
	# 	plt.scatter(XTest, yTest,  color='black')
	# 	plt.plot(XTest, regr.predict(XTest), color='blue',
	# 	         linewidth=3)

	# 	plt.xticks(())
	# 	plt.yticks(())

	# 	plt.show()

	# 	plt.scatter(X, y,  color='black')
	# 	plt.plot(X, regr.predict(X), color='blue',
	# 	         linewidth=3)

	# 	plt.xticks(())
	# 	plt.yticks(())

	# 	plt.show()

	y_true = yTest
	y_pred = regr.predict(XTest)
	print "Explained variance: " + str(explained_variance_score(y_true, y_pred))
	print "R^2 Library: " + str(r2_score(y_true, y_pred))

def convert_string_list_to_float_list(string_list):
	float_list = []
	for s in string_list:
		float_list.append(float(s))
	return float_list		

def main(args):
	train_file = open('../Data/Train/train_final.txt')
	test_file = open('../Data/Test/test_final.txt')
	feature_idx = int(args[1])

	X = []; y_hi = []; y_lo = [];
	result_lo_idx = 0
	result_hi_idx = 1
	if args[0] == '-n':
		result_lo_idx = 2
		result_hi_idx = 3

	for line in train_file:
		arr = line.split()
		if feature_idx >= 1 and (feature_idx < len(arr) - 4):
			features = arr[feature_idx: feature_idx + 1]
		elif feature_idx == 12:
			features = arr[1: 3]
		else:
			features = arr[1:(len(arr)) - 4]
		
		results = arr[(len(arr)) - 4:]
		X.append(convert_string_list_to_float_list(features))
		y_hi.append(float(results[result_hi_idx]))
		y_lo.append(float(results[result_lo_idx]))

	X_test = []; y_hi_test = []; y_lo_test = []
	for line in test_file:
		arr = line.split()
		# features = arr[1:(len(arr)) - 4]
		if feature_idx >= 1 and (feature_idx < len(arr) - 4):
			features = arr[feature_idx: feature_idx + 1]
		elif feature_idx == 12:
			features = arr[1: 3]
		else:
			features = arr[1:(len(arr)) - 4]
		# print features
		results = arr[(len(arr)) - 4:]
		X_test.append(convert_string_list_to_float_list(features))
		y_hi_test.append(float(results[result_hi_idx]))
		y_lo_test.append(float(results[result_lo_idx]))


	X_array = np.array(X)
	y_hi_array = np.array(y_hi)
	X_test_array = np.array(X_test)
	y_hi_test_array =  np.array(y_hi_test)

	print "High G: \n"
	Linear_Regression(X, y_hi, X_test, y_hi_test)
	SVM(X, y_hi, X_test, y_hi_test)
	KNN(X, y_hi, X_test, y_hi_test)
	DT(X, y_hi, X_test, y_hi_test)

	print "----------------"

	print "Low G: \n"
	Linear_Regression(X, y_lo, X_test, y_lo_test)
	SVM(X, y_lo, X_test, y_lo_test)
	KNN(X, y_lo, X_test, y_lo_test)
	DT(X, y_lo, X_test, y_lo_test)



if __name__ == "__main__":
    main(sys.argv[1:])