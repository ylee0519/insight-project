#!/usr/bin/python

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pdb

from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc

startups = pd.read_csv('5yo_startups_cleaned.csv')

# Pre-processing
active_startups = startups.drop(startups.index[startups.stage == 'Exited'])
y = active_startups.stage.isin(['A', 'B', 'C', 'Late']).astype(int)
data = active_startups.drop(['business_models', 'employee_count', 'est_founding_date', 'funding', 'id', 'industries', 'location', 'name', 'stage', 'stories', 'total_funding', 'website'], axis=1)

Xp, yp = data[y == 1], y[y == 1]
Xn, yn = data[y == 0], y[y == 0]

Xp_train, Xp_test, yp_train, yp_test = train_test_split(Xp, yp, test_size=0.4, random_state=0)
Xn_train, Xn_test, yn_train, yn_test = train_test_split(Xn, yn, test_size=0.4, random_state=0)

X_train = np.concatenate([Xp_train, Xn_train], axis=0)
y_train = np.concatenate([yp_train, yn_train], axis=0)

X_test = np.concatenate([Xp_test, Xn_test], axis=0)
y_test = np.concatenate([yp_test, yn_test], axis=0)

clf = RandomForestClassifier(n_jobs=2)
clf = clf.fit(X_train, y_train)

y_pred = clf.predict_proba(X_test)[:, 1]

fpr, tpr, _ = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure()
plt.style.use('fivethirtyeight')
plt.plot(fpr, tpr, 'b-', linewidth=4, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--', linewidth=2)
plt.xlim([-0.01, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Random forest model')
plt.legend(loc="lower right")
plt.savefig('roc_curve.png', dpi=400, bbox_inches='tight')

# Evalute the importance of features
importance = clf.feature_importances_
indices = np.argsort(importance)[::-1]

# Print feature ranking
print("Feature ranking:")
for f in range(10):
    print("%2d. %s (%f)" % (f + 1, data.columns[indices][f], importance[indices[f]]))

plt.figure()
plt.style.use('fivethirtyeight')
plt.title("Feature importances", fontsize=20)
plt.barh(range(10), importance[indices[:10]][::-1], align="center")
plt.yticks(range(10), data.columns[indices][:10][::-1], fontsize=16)
plt.ylim([-1, 10])
plt.tight_layout()
plt.savefig('feature_importances.png', dpi=400, bbox_inches='tight')