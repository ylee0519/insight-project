#!/usr/bin/python

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pdb

from datetime import datetime, timedelta
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc

# Pre-processing
startups = pd.read_csv('5yo_startups_cleaned.csv')

active_startups = startups.drop(startups.index[startups.stage == 'Exited'])
active_startups['age'] = pd.to_datetime(datetime(2015, 1, 1)) - pd.to_datetime(active_startups.est_founding_date)

# Use two-year-olds as test set
toddlers = active_startups[active_startups.age > timedelta(365*2)].copy()
two_year_olds = active_startups[active_startups.age <= timedelta(365*2)].copy()

print len(active_startups), len(toddlers), len(two_year_olds)
# pdb.set_trace()

y_train = toddlers.stage.isin(['A', 'B', 'C', 'Late']).astype(int)
X_train = toddlers.drop([
    'age',
    'business_models', 
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
random_forest_clf = RandomForestClassifier(n_jobs=2)
random_forest_clf = random_forest_clf.fit(X_train, y_train)

y_pred_random_forest = random_forest_clf.predict_proba(X_test)[:, 1]

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
logistic_clf = LogisticRegression()
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

