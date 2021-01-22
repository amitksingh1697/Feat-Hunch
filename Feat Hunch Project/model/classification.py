from __future__ import print_function
from sklearn import preprocessing, decomposition, svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import *
import random
import numpy as np
import matplotlib.pylab as pl
import pandas as pd

clfs = {'LR': LogisticRegression(penalty='l1', C=1e5),
        'SVM': svm.SVC(kernel='linear', probability=True, random_state=0),
        'NB': GaussianNB()
        }


class Model:

    def __init__(self, dataSet, dependentVar, doFeatureSelection=True, doPCA=False, nComponents=10):
        for i,tp in enumerate(dataSet.dtypes):
            if tp == 'object':
                print ('Encoding feature \"' + dataSet.columns[i] + '\" ...')
                print ('Old dataset shape: ' + str(dataSet.shape))
                temp = pd.get_dummies(dataSet[dataSet.columns[i]],prefix=dataSet.columns[i])
                dataSet = pd.concat([dataSet,temp],axis=1).drop(dataSet.columns[i],axis=1)
                print ('New dataset shape: ' + str(dataSet.shape))



        y = dataSet.loc[:,dependentVar]
        labels = preprocessing.LabelEncoder().fit_transform(y)
        X = dataSet.drop(dependentVar,1).values
        if doFeatureSelection:
            print ('Performing Feature Selection:')
            print ('Shape of dataset before feature selection: ' + str(X.shape))
            clf = DecisionTreeClassifier(criterion='entropy')
            #X = clf.fit(X, y).transform(X)
            print ('Shape of dataset after feature selection: ' + str(X.shape) + '\n')

        # Normalize values
        X = preprocessing.StandardScaler().fit(X).transform(X)

        # Collapse features using principal component analysis
        if doPCA:
            print ('Performing PCA')
            estimator = decomposition.PCA(n_components=nComponents)
            X = estimator.fit_transform(X)
            print ('Shape of dataset after PCA: ' + str(X.shape) + '\n')

        # Save processed dataset, labels and student ids
        self.dataset = X
        self.labels = labels
        self.students = dataSet.index


    def subsample(self, x, y, ix, subsample_ratio=1.0):

        # Get indexes of instances that belong to classes 0 and 1
        indexes_0 = [item for item in ix if y[item] == 0]
        indexes_1 = [item for item in ix if y[item] == 1]

        # Determine how large the new majority class set should be
        sample_length = int(len(indexes_1)*subsample_ratio)
        sample_indexes = random.sample(indexes_0, sample_length) + indexes_1

        return sample_indexes


    def runClassification(self, outputFormat='score', doSubsampling=False, subRate=1.0,
                            doSMOTE=False, pctSMOTE=100, nFolds=10, models=['LR'], topK=.1,):

        # Return a simple overall accuracy score
        if outputFormat=='score':
            if doSMOTE or doSubsampling:
                print ('Sorry, scoring with subsampling or SMOTE not yet implemented')
                return
            # Iterate through each classifier to be evaluated
            for ix,clf in enumerate([clfs[x] for x in models]):
                kf = cross_validation.KFold(len(self.dataset), nFolds, shuffle=True)
                scores = cross_validation.cross_val_score(clf, self.dataset, self.labels, cv=kf)
                print (models[ix]+ ' Accuracy: %.2f' % np.mean(scores))

        # Return a summary table describing several metrics or a confusion matrix
        elif outputFormat=='summary' or outputFormat=='matrix':
            for ix,clf in enumerate([clfs[x] for x in models]):
                # Store the prediction results and their corresponding real labels for each fold
                y_prediction_results = []; y_smote_prediction_results = []
                y_original_values = []

                # Generate indexes for the K-fold setup
                kf = cross_validation.StratifiedKFold(self.labels, n_folds=nFolds)
                for i, (train, test) in enumerate(kf):
                    if doSubsampling:
                    	# Remove some random majority class instances to balance data
                        train = self.subsample(self.dataset,self.labels,train,subRate)
                    if doSMOTE:
                        # SMOTE the minority class and append new instances to training set
                        minority = self.dataset[train][np.where(self.labels[train]==1)]
                        smotted = self.SMOTE(minority, pctSMOTE, 5)
                        X_train_smote = np.vstack((self.dataset[train],smotted))
                        y_train_smote = np.append(self.labels[train],np.ones(len(smotted),dtype=np.int32))
                        # Fit the new training set to selected model
                        y_pred_smote = clf.fit(X_train_smote, y_train_smote).predict(self.dataset[test])
                        # Generate SMOTEd predictions and append that to the rersults list
                        y_smote_prediction_results = np.concatenate((y_smote_prediction_results,y_pred_smote),axis=0)

                    # Generate predictions for current hold-out sample in i-th fold
		    #fitted_clf = clf.fit(self.dataset[train], self.labels[train])
		    # self.feature_importances = getattr(fitted_clf, 'feature_importances_', None)
                    y_pred = fitted_clf.predict(self.dataset[test])
                    # Append results to previous ones
                    y_prediction_results = np.concatenate((y_prediction_results,y_pred),axis=0)
                    # Store the corresponding original values for the predictions just generated
                    y_original_values = np.concatenate((y_original_values,self.labels[test]),axis=0)

                # Print result summary table based on k-fold
                # This is specific to our particular experiment and classes are hard coded
                # When oversampling is True, both results are displayed
                if outputFormat=='summary':
                    print ('\t\t\t\t\t\t'+models[ix]+ ' Summary Results')
                    cm = classification_report(y_original_values, y_prediction_results,target_names=['Graduated','Did NOT Graduate'])
                    print(str(cm)+'\n')
                    if doSMOTE:
                        print ('\t\t\t\t\t\t'+models[ix]+ ' SMOTE Summary Results')
                        cm = classification_report(y_original_values, y_smote_prediction_results,target_names=['Graduated','Did NOT Graduate'])
                        print(str(cm)+'\n')
                    print ('----------------------------------------------------------\n')

                # Print the confusion matrix
                else:
                    print ('\t\t\t\t\t'+models[ix]+ ' Confusion Matrix')
                    print ('\t\t\t\tGraduated\tDid NOT Graduate')
                    cm = confusion_matrix(y_original_values, y_prediction_results)
                    print ('Graduated\t\t\t%d\t\t%d'% (cm[0][0],cm[0][1]))
                    print ('Did NOT Graduate\t%d\t\t%d'% (cm[1][0],cm[1][1]))
                    if doSMOTE:
                        print ('\n\t\t\t\t'+models[ix]+ ' SMOTE Confusion Matrix')
                        print ('\t\t\t\tGraduated\tDid NOT Graduate')
                        cm = confusion_matrix(y_original_values, y_smote_prediction_results)
                        print ('Graduated\t\t\t%d\t\t%d'% (cm[0][0],cm[0][1]))
                        print ('Did NOT Graduate\t%d\t\t%d'% (cm[1][0],cm[1][1]))

                    print ('----------------------------------------------------------\n')

        # Generate ROC curves
        # The majority of the structure here is similar to above, so refer to early comments
        # TODO: create consise procedures to avoid code duplication
        elif outputFormat=='roc':
            for ix,clf in enumerate([clfs[x] for x in models]):
                kf = cross_validation.StratifiedKFold(self.labels, n_folds=nFolds)
                mean_tpr = mean_smote_tpr = 0.0
                mean_fpr = mean_smote_fpr = np.linspace(0, 1, 100)

                for i, (train, test) in enumerate(kf):
                    if doSubsampling:
                        train = self.subsample(self.dataset,self.labels,train,subRate)
                    if doSMOTE:
                        minority = self.dataset[train][np.where(self.labels[train]==1)]
                        smotted = self.SMOTE(minority, pctSMOTE, 5)
                        X_train = np.vstack((self.dataset[train],smotted))
                        y_train = np.append(self.labels[train],np.ones(len(smotted),dtype=np.int32))
                        probas2_ = clf.fit(X_train, y_train).predict_proba(self.dataset[test])
                        fpr, tpr, thresholds = roc_curve(self.labels[test], probas2_[:, 1])
                        mean_smote_tpr += np.interp(mean_smote_fpr, fpr, tpr)
                        mean_smote_tpr[0] = 0.0

                    # Generate "probabilities" for the current hold out sample being predicted
		    #fitted_clf = clf.fit(self.dataset[train], self.labels[train])
		    # self.feature_importances = getattr(fitted_clf, 'feature_importances_', None)
                    probas_ = fitted_clf.predict_proba(self.dataset[test])
                    # Compute ROC curve and area the curve
                    fpr, tpr, thresholds = roc_curve(self.labels[test], probas_[:, 1])
                    mean_tpr += np.interp(mean_fpr, fpr, tpr)

                # Plot ROC baseline
                pl.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Baseline')

                # Compute true positive rates
                mean_tpr /= len(kf)
                mean_tpr[-1] = 1.0
                mean_auc = auc(mean_fpr, mean_tpr)

                # Plot results
                pl.plot(mean_fpr, mean_tpr, 'k-',
                        label='Mean ROC (area = %0.2f)' % mean_auc, lw=2)

                # Plot results with oversampling
                if doSMOTE:
                    mean_smote_tpr /= len(kf)
                    mean_smote_tpr[-1] = 1.0
                    mean_smote_auc = auc(mean_smote_fpr, mean_smote_tpr)
                    pl.plot(mean_smote_fpr, mean_smote_tpr, 'r-',
                        label='Mean smote ROC (area = %0.2f)' % mean_smote_auc, lw=2)

                pl.xlim([-0.05, 1.05])
                pl.ylim([-0.05, 1.05])
                pl.xlabel('False Positive Rate')
                pl.ylabel('True Positive Rate')
                pl.title(models[ix]+ ' ROC')
                pl.legend(loc="lower right")
                pl.show()

        # Generate Precision-Recall curves, topK precision, or list of topK at risk scores
        elif outputFormat =='prc' or outputFormat =='topk' or outputFormat =='risk':
            for ix,clf in enumerate([clfs[x] for x in models]):
                y_prob = []; y_smote_prob = []
                y_prediction_results = []; y_smote_prediction_results = []
                y_original_values = []; test_indexes = []

                kf = cross_validation.StratifiedKFold(self.labels, n_folds=nFolds, shuffle=True)
                mean_pr = mean_smote_pr = 0.0
                mean_rc = mean_smote_rc = np.linspace(0, 1, 100)

                for i, (train, test) in enumerate(kf):
                    if doSubsampling:
                        train = self.subsample(self.dataset,self.labels,train,subRate)

                    if doSMOTE:
                        clf2 = clf
                        minority = self.dataset[train][np.where(self.labels[train]==1)]
                        smotted = self.SMOTE(minority, pctSMOTE, 5)
                        X_train = np.vstack((self.dataset[train],smotted))
                        y_train = np.append(self.labels[train],np.ones(len(smotted),dtype=np.int32))
                        clf2.fit(X_train, y_train)
                        probas2_ = clf2.predict_proba(self.dataset[test])
                        y_pred_smote = clf2.predict(self.dataset[test])
                        # Generate SMOTEd predictions and append that to the rersults list
                        y_smote_prediction_results = np.concatenate((y_smote_prediction_results,y_pred_smote),axis=0)
                        y_smote_prob = np.concatenate((y_smote_prob,probas2_[:, 1]),axis=0)

                    clf.fit(self.dataset[train], self.labels[train])
                    y_pred = clf.predict(self.dataset[test])
                    y_prediction_results = np.concatenate((y_prediction_results,y_pred),axis=0)
                    test_indexes = np.concatenate((test_indexes,test),axis=0)
                    y_original_values = np.concatenate((y_original_values,self.labels[test]),axis=0)
                    probas_ = clf.predict_proba(self.dataset[test])
                    y_prob = np.concatenate((y_prob,probas_[:, 1]),axis=0)

                # Compute overall prediction, recall and area under PR-curve
                precision, recall, thresholds = precision_recall_curve(y_original_values, y_prob)
                pr_auc = auc(recall, precision)


                if doSMOTE:
                    precision_smote, recall_smote, thresholds_smote = precision_recall_curve(y_original_values, y_smote_prob)
                    pr_auc_smote = auc(recall_smote, precision_smote)

                # Output the precision recall curve
                if outputFormat=='prc':
                    pl.plot(recall, precision, color = 'b', label='Precision-Recall curve (area = %0.2f)' % pr_auc)
                    if doSMOTE:
                        pl.plot(recall_smote, precision_smote, color = 'r', label='SMOTE Precision-Recall curve (area = %0.2f)' % pr_auc_smote)
                    pl.xlim([-0.05, 1.05])
                    pl.ylim([-0.05, 1.05])
                    pl.xlabel('Recall')
                    pl.ylabel('Precision')
                    pl.title(models[ix]+ ' Precision-Recall')
                    pl.legend(loc="lower right")
                    pl.show()

                # Output a list of the topK% students at highest risk along with their risk scores
                elif outputFormat =='risk':
                    test_indexes = test_indexes.astype(int)
                    sort_ix = np.argsort(test_indexes)
                    students_by_risk = self.students[test_indexes]
                    y_prob = ((y_prob[sort_ix])*100).astype(int)
                    probas = np.column_stack((students_by_risk,y_prob))
                    r = int(topK*len(y_original_values))
                    print (models[ix]+ ' top ' + str(100*topK) + '%' + ' highest risk')
                    print ('--------------------------')
                    print ('%-15s %-10s' % ('Student','Risk Score'))
                    print ('%-15s %-10s' % ('-------','----------'))
                    probas = probas[np.argsort(probas[:, 1])[::-1]]
                    for i in range(r):
                        print ('%-15s %-10d' % (probas[i][0], probas[i][1]))
                    print ('\n')

                # Output the precision on the topK%
                else:
                    ord_prob = np.argsort(y_prob,)[::-1]
                    r = int(topK*len(y_original_values))
                    print (models[ix]+ ' Precision at top ' + str(100*topK) + '%')
                    print (np.sum(y_original_values[ord_prob][:r])/r)

                    if doSMOTE:
                        ord_prob = np.argsort(y_smote_prob,)[::-1]
                        print (models[ix]+ ' SMOTE Precision at top ' + str(100*topK) + '%')
                        print (np.sum(y_original_values[ord_prob][:r])/r)
                    print ('\n')
