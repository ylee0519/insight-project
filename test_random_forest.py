#!/usr/bin/python

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pdb

from datetime import datetime, timedelta
from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
from sklearn.grid_search import GridSearchCV

# Pre-processing
startups = pd.read_csv('5yo_startups_cleaned.csv')

active_startups = startups.drop(startups.index[startups.stage == 'Exited'])
active_startups['age'] = (pd.to_datetime(datetime(2015, 1, 1)) - pd.to_datetime(active_startups.est_founding_date)).astype('timedelta64[D]')
active_startups['employee_growth'] = active_startups['employee'].div(active_startups['age'])

# Use two-year-olds as test set
toddlers = active_startups[active_startups.age > (365*2)].copy()
two_year_olds = active_startups[active_startups.age <= (365*2)].copy()

print len(active_startups), len(toddlers), len(two_year_olds)
pdb.set_trace()

y_train = toddlers.stage.isin(['A', 'B', 'C', 'Late']).astype(int)
X_train = toddlers.drop([
    'age',
    'business_models',
    'employee', 
    'employee_count',
    'est_founding_date', 
    'funding', 
    'id', 
    'industries', 
    'location', 
    'name',
    'stage', 
    'stories', 
    'total_funding', 
    'website'], 
    axis=1)

y_test = two_year_olds.stage.isin(['A', 'B', 'C', 'Late']).astype(int)
X_test = two_year_olds.drop([
    'age',
    'business_models',
    'employee', 
    'employee_count', 
    'est_founding_date', 
    'funding', 
    'id', 
    'industries', 
    'location', 
    'name',
    'stage', 
    'stories', 
    'total_funding', 
    'website'], 
    axis=1)

# Try random forest
# tuned_parameters = [{'max_features': ['sqrt', 'log2'], 'n_estimators': [10, 30, 100, 300, 1000]}]
# random_forest_clf = GridSearchCV( RandomForestClassifier(min_samples_split=1, n_jobs=-1), tuned_parameters, cv=5 )
# random_forest_clf = random_forest_clf.fit(X_train, y_train)
# print random_forest_clf.best_estimator_
# pdb.set_trace()

random_forest_clf = RandomForestClassifier(min_samples_split=1, max_features='lolg2', n_estimators=1000, n_jobs=-1)
print np.mean(cross_val_score(random_forest_clf, X_train, y_train, cv=5))
pdb.set_trace()
random_forest_clf = random_forest_clf.fit(X_train, y_train)

y_pred_random_forest = random_forest_clf.predict_proba(X_test)[:, 1]
# print random_forest_clf.score(X_test, y_test)

# Evalute the importance of features
importance = random_forest_clf.feature_importances_
indices = np.argsort(importance)[::-1]

# Print feature ranking
print("Feature ranking:")
for f in range(10):
    print("%2d. %s (%f)" % (f + 1, X_train.columns[indices][f], importance[indices[f]]))

plt.figure()
plt.style.use('fivethirtyeight')
plt.title("Feature importances", fontsize=20)
plt.barh(range(10), importance[indices[:10]][::-1], align="center")
plt.yticks(range(10), X_train.columns[indices][:10][::-1], fontsize=16)
plt.ylim([-1, 10])
plt.tight_layout()
# plt.show()
plt.savefig('feature_importances.png', dpi=400, bbox_inches='tight')

# Try logistic regression
tuned_params = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000] }
logistic_clf = GridSearchCV(LogisticRegression(penalty='l1'), tuned_params, cv=5) 
logistic_clf = logistic_clf.fit(X_train, y_train)

y_pred_logistic = logistic_clf.predict_proba(X_test)[:, 1]

# Make ROC curve
fpr_random_forest, tpr_random_forest, _ = roc_curve(y_test, y_pred_random_forest)
roc_auc_random_forest = auc(fpr_random_forest, tpr_random_forest)

fpr_logistic, tpr_logistic, _ = roc_curve(y_test, y_pred_logistic)
roc_auc_logistic = auc(fpr_logistic, tpr_logistic)

plt.figure()
plt.style.use('fivethirtyeight')
plt.plot(fpr_random_forest, tpr_random_forest, 'b-', linewidth=4, label='Random forest (AUC = %0.2f)' % roc_auc_random_forest)
plt.plot(fpr_logistic, tpr_logistic, 'r-', linewidth=4, label='Logistic regression (AUC = %0.2f)' % roc_auc_logistic)
plt.plot([0, 1], [0, 1], 'k--', linewidth=2)
plt.xlim([-0.01, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Random forest vs. Logistic regression')
plt.legend(loc="lower right")
# plt.show()
plt.savefig('roc_curve.png', dpi=400, bbox_inches='tight')

