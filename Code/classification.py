import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

def analysis_print(name, per_correct, prec, recall, beta):
	print name + ": "
	print "Percenatage Correct: %f" % per_correct
	print "Precision: %f" % prec 
	print "Recall: %f" % recall
	print "FBeta %f" % beta

def error(yPred, yTest):
    testsize = len(yTest)
    correctIndices = []
    correct = 0.0
    for i in range(testsize):
        if yTest[i] == yPred[i]:
            correct = correct + 1
            correctIndices.append(i)
    per_correct = correct / testsize
    return (correctIndices, per_correct)

def precision_recall(y_pred, y_true):
	from sklearn.metrics import precision_score
	from sklearn.metrics import recall_score
	from sklearn.metrics import fbeta_score
	precision = precision_score(y_true, y_pred)
	recall = recall_score(y_true, y_pred)
	beta = fbeta_score(y_true, y_pred,beta=0.5)
	return (precision, recall, beta)


def ROC(cmetric, yTest):
    fpr, tpr, thresholds = roc_curve(yTest, cmetric[:, 1])
    roc_auc = auc(fpr, tpr)
    print "Area under the ROC curve : %f" % roc_auc

    #Plotting the ROC
    # pl.clf()
    # pl.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    # pl.plot([0, 1], [0, 1], 'k--')
    # pl.xlim([0.0, 1.0])
    # pl.ylim([0.0, 1.0])
    # pl.xlabel('False Positive Rate')
    # pl.ylabel('True Positive Rate')
    # pl.title('Receiver operating characteristic example')
    # pl.legend(loc="lower right")
    # pl.show()

    return (fpr, tpr)

def SVM(X, y, XTest, yTest):
	from sklearn.svm import SVC
	clf = SVC()	
	yPred = clf.fit(X,y).predict(XTest)

	(correctIndices, per_correct) = error(yPred, yTest)
	(prec, recall, beta) = precision_recall(yPred, yTest)
	analysis_print("SVM", per_correct, prec, recall, beta)
	print

def GaussianNB(X, y, XTest, yTest):
	from sklearn.naive_bayes import GaussianNB
	clf = GaussianNB()	
	probas = clf.fit(X, y).predict_proba(XTest)
	yPred = clf.fit(X,y).predict(XTest)

	(correctIndices, per_correct) = error(yPred, yTest)
	(prec, recall, beta) = precision_recall(yPred, yTest)
	analysis_print("Gaussian Naive Bayes", per_correct, prec, recall, beta)
	fpr, tpr = ROC(probas, yTest)
	print

def DecisionTree(X, y, XTest, yTest):
	from sklearn.tree import DecisionTreeClassifier
	clf = DecisionTreeClassifier()	
	probas = clf.fit(X, y).predict_proba(XTest)
	yPred = clf.fit(X,y).predict(XTest)

	(correctIndices, per_correct) = error(yPred, yTest)
	(prec, recall, beta) = precision_recall(yPred, yTest)
	analysis_print("DecisionTree", per_correct, prec, recall, beta)
	fpr, tpr = ROC(probas, yTest)
	print
	

def RandomForest(X, y, XTest, yTest):
	from sklearn.ensemble import RandomForestClassifier
	clf = RandomForestClassifier()	
	probas = clf.fit(X, y).predict_proba(XTest)
	yPred = clf.fit(X,y).predict(XTest)

	(correctIndices, per_correct) = error(yPred, yTest)
	(prec, recall, beta) = precision_recall(yPred, yTest)
	analysis_print("RandomForest", per_correct, prec, recall, beta)
	fpr, tpr = ROC(probas, yTest)
	print

def KNeighborsClassifier(X,y,XTest, yTest):
	from sklearn.neighbors import KNeighborsClassifier
	neigh = KNeighborsClassifier(n_neighbors=3)
	yPred =	neigh.fit(X, y).predict(XTest)
	probas = neigh.fit(X, y).predict_proba(XTest)

	(correctIndices, per_correct) = error(yPred, yTest)
	(prec, recall, beta) = precision_recall(yPred, yTest)
	analysis_print("KNeighbors", per_correct, prec, recall, beta)
	fpr, tpr = ROC(probas, yTest)
	print
	

def treeFeatureImportance(X,y):
	from sklearn.ensemble import ExtraTreesClassifier
	forest = ExtraTreesClassifier(n_estimators=250,
                              random_state=0)

	forest.fit(X, y)
	importances = forest.feature_importances_
	std = np.std([tree.feature_importances_ for tree in forest.estimators_],
             axis=0)
	indices = np.argsort(importances)[::-1]

	# Print the feature ranking
	print("Feature ranking:")

	for f in range(X.shape[1]):
		print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

	# Plot the feature importances of the forest
	plt.figure()
	plt.title("Feature importances")
	plt.bar(range(X.shape[1]), importances[indices],
	       color="r", yerr=std[indices], align="center")
	plt.xticks(range(X.shape[1]), indices)
	plt.xlim([-1, X.shape[1]])
	plt.show()

def convert_string_list_to_float_list(string_list):
	float_list = []
	for s in string_list:
		float_list.append(float(s))
	return float_list		

def main(args):
	train_file = open('../Data/Train/train_final_bin.txt')
	test_file = open('../Data/Test/test_final_bin.txt')

	X = []; y_hi = []; y_lo = [];
	result_lo_idx = 0
	result_hi_idx = 1
	if args[0] == '-n':
		result_lo_idx = 2
		result_hi_idx = 3

	for line in train_file:
		arr = line.split()
		features = arr[1:(len(arr)) - 4]
		results = arr[(len(arr)) - 4:]
		print results
		X.append(convert_string_list_to_float_list(features))
		y_hi.append(int(results[result_hi_idx]))
		y_lo.append(int(results[result_lo_idx]))

	X_test = []; y_hi_test = []; y_lo_test = []
	for line in test_file:
		arr = line.split()
		features = arr[1:(len(arr)) - 4]
		results = arr[(len(arr)) - 4:]
		X_test.append(convert_string_list_to_float_list(features))
		y_hi_test.append(int(results[result_hi_idx]))
		y_lo_test.append(int(results[result_lo_idx]))
	

	X_array = np.array(X)
	y_hi_array = np.array(y_hi)
	X_test_array = np.array(X_test)
	y_hi_test_array =  np.array(y_hi_test)

	print "High G: \n"
	GaussianNB(X, y_lo, X_test, y_lo_test)
	DecisionTree(X_array, y_hi_array, X_test_array, y_hi_test_array)
	KNeighborsClassifier(X_array, y_hi_array, X_test_array, y_hi_test_array)
	RandomForest(X, y_lo, X_test, y_lo_test)
	SVM(X, y_lo, X_test, y_lo_test)
	treeFeatureImportance(np.array(X), np.array(y_hi))	

	print "----------------"

	print "Low G: \n"
	GaussianNB(X, y_lo, X_test, y_lo_test)
	DecisionTree(X, y_lo, X_test, y_lo_test)
	KNeighborsClassifier(X,y_lo, X_test, y_lo_test)
	RandomForest(X, y_lo, X_test, y_lo_test)
	SVM(X, y_lo, X_test, y_lo_test)
	treeFeatureImportance(np.array(X), np.array(y_lo))



if __name__ == "__main__":
    main(sys.argv[1:])