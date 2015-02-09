#!/usr/bin/python

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pdb

from datetime import datetime, timedelta
from sklearn import preprocessing
from sklearn import svm
from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
from sklearn.grid_search import GridSearchCV

def evaluate_features(random_forest_clf, columns, path_to_fig):
    # Evalute the importance of features
    importance = random_forest_clf.feature_importances_
    indices = np.argsort(importance)[::-1]

    # Print feature ranking
    print "Feature ranking:"
    for f in range(10):
        print("%2d. %s (%f)" % (f + 1, columns[indices][f], importance[indices[f]]))

    plt.figure()
    plt.style.use('ggplot')
    plt.title("Feature importances", fontsize=18)
    plt.barh(range(10), importance[indices[:10]][::-1], align="center", color='orange', alpha=0.8)
    plt.yticks(range(10), columns[indices][:10][::-1], fontsize=16)
    plt.ylim([-1, 10])
    # plt.show()
    plt.savefig(path_to_fig, dpi=400, bbox_inches='tight')

def standardize_data(X_train, X_cv, i_cols):
    X_train_scaled = X_train.copy()
    X_train_scaled[:, i_cols] = preprocessing.scale(X_train[:, i_cols])
    scaler = preprocessing.StandardScaler().fit(X_train[:, i_cols])
    X_cv_scaled = X_cv.copy()
    X_cv_scaled[:, i_cols] = scaler.transform(X_cv[:, i_cols])
    print 'Standardization is DONE'

    return X_train_scaled, X_cv_scaled

def main(X, y, ver):
    # Try random forest
    random_forest_clf = RandomForestClassifier(n_estimators=1000, n_jobs=-1)
    cv_scores = cross_val_score(random_forest_clf, X, y, cv=5, n_jobs=2)
    print 'Accuracy: Mean = %.2f, Std = %.2f' % (np.mean(cv_scores), np.std(cv_scores))

    random_forest_clf = random_forest_clf.fit(X, y)
    path_to_fig = 'feature_importances_v%d.png' % ver
    evaluate_features(random_forest_clf, X.columns, path_to_fig)

    # 90% for training and 10% for validation
    X_train, X_cv, y_train, y_cv = train_test_split(X, y, test_size=0.1, random_state=1)

    random_forest_clf = RandomForestClassifier(n_estimators=1000, n_jobs=-1).fit(X_train, y_train)
    y_pred_random_forest = random_forest_clf.predict_proba(X_cv)[:, 1]

    # Standardize numerical features
    X_train_scaled, X_cv_scaled = standardize_data(X_train, X_cv, [0, 1])

    # Try logistic regression
    tuned_params = {'C': [0.01, 0.1, 1, 10, 100]}
    logistic_clf = GridSearchCV(LogisticRegression(), tuned_params, cv=5)
    logistic_clf = logistic_clf.fit(X_train_scaled, y_train)
    y_pred_logistic = logistic_clf.predict_proba(X_cv_scaled)[:, 1]
    
    # Try support vector machines
    svc = GridSearchCV(svm.SVC(probability=True), tuned_params, cv=5)
    svc = svc.fit(X_train_scaled, y_train)
    y_pred_svc = svc.predict_proba(X_cv_scaled)[:, 1]

    # Make ROC curves
    fpr_random_forest, tpr_random_forest, _ = roc_curve(y_cv, y_pred_random_forest)
    roc_auc_random_forest = auc(fpr_random_forest, tpr_random_forest)

    fpr_logistic, tpr_logistic, _ = roc_curve(y_cv, y_pred_logistic)
    roc_auc_logistic = auc(fpr_logistic, tpr_logistic)

    fpr_svc, tpr_svc, _ = roc_curve(y_cv, y_pred_svc)
    roc_auc_svc = auc(fpr_svc, tpr_svc)

    plt.figure()
    plt.style.use('ggplot')

    plt.plot(fpr_random_forest, tpr_random_forest, linewidth=3, label='Random forest (AUC = %0.2f)' % roc_auc_random_forest)
    plt.plot(fpr_logistic, tpr_logistic, linewidth=3, label='Logistic regression (AUC = %0.2f)' % roc_auc_logistic)
    plt.plot(fpr_svc, tpr_svc, linewidth=3, label='Support vector machines (AUC = %0.2f)' % roc_auc_svc)

    plt.plot([0, 1], [0, 1], 'k--', linewidth=1.5)
    plt.xlim([-0.01, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Model comparison')
    plt.legend(loc="lower right")

    path_to_fig = 'roc_curve_v%d.png' % ver
    plt.savefig(path_to_fig, dpi=400, bbox_inches='tight')


if __name__ == '__main__':
    # Pre-processing
    startups = pd.read_csv('5yo_startups_cleaned.csv')
    active_startups = startups.drop(startups.index[startups.stage == 'Exited'])
    age = (pd.to_datetime(datetime(2015, 1, 1)) - pd.to_datetime(active_startups.est_founding_date)).astype('timedelta64[D]')

    # Use two-year-olds as test set
    toddlers = active_startups[age > (365*2)].copy()
    two_year_olds = active_startups[age <= (365*2)].copy()
    print len(active_startups), len(toddlers), len(two_year_olds)

    y = toddlers.stage.isin(['A', 'B', 'C', 'Late']).astype(int)
    X = toddlers.drop([
        'business_models',
        'est_founding_date',
        'employee_count',
        'funding', 
        'id', 
        'industries', 
        'location', 
        'name',
        'stage', 
        'stories',
        'website',
        'series_A_funding_date',
        'angel funding'], 
        axis=1)

    main(X, y, 1)

    # Test a second model: replace pre-series A funding with angel funding
    y = toddlers.stage.isin(['A', 'B', 'C', 'Late']).astype(int)
    X = toddlers.drop([
        'business_models',
        'est_founding_date',
        'employee_count',
        'funding', 
        'id', 
        'industries', 
        'location', 
        'name',
        'stage', 
        'stories',
        'website',
        'series_A_funding_date',
        'pre-series-A funding'], 
        axis=1)

    main(X, y, 2)
