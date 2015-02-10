#!/usr/bin/python

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pdb

from datastore import local
from datetime import datetime, timedelta

from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import RandomForestClassifier

def populate_prediction_table(startups):
    db = local()
    q = '''DROP TABLE IF EXISTS prediction;'''
    db.cur.execute(q)

    q = '''CREATE TABLE prediction (Id INT, Score INT);'''
    db.cur.execute(q)

    for i, row in startups.iterrows():
        q = '''INSERT INTO prediction VALUES (%s, %s);'''
        db.cur.execute(q, (row.id, row.score))

    db.con.commit()
    db.close()

if __name__ == '__main__':
    # Pre-processing of data
    startups = pd.read_csv('5yo_startups_cleaned.csv')

    active_startups = startups.drop(startups.index[startups.stage == 'Exited'])
    age = (pd.to_datetime(datetime(2015, 1, 1)) - pd.to_datetime(active_startups.est_founding_date)).astype('timedelta64[D]')

    toddlers = active_startups[age > (365*2)].copy()
    two_year_olds = active_startups[age <= (365*2)].copy()

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

    clf = RandomForestClassifier(n_estimators=1000, n_jobs=-1).fit(X, y)

    y_test = two_year_olds.stage.isin(['A', 'B', 'C', 'Late']).astype(int)
    X_test = two_year_olds.drop([
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

    print clf.score(X_test, y_test)

    pred = 100*clf.predict_proba(X_test)

    two_year_olds[u'score'] = pd.Series(pred[:, 1].astype(int), index=two_year_olds.index)
    two_year_olds[u'pred'] = pd.Series(clf.predict(X_test), index=two_year_olds.index)

    n_startups = np.zeros((78, 1))
    funding_rate = np.zeros((78, 1))

    for i, x in enumerate(two_year_olds.columns[28:106].values):
        n_startups[i] = two_year_olds[x].sum()
        funding_rate[i] = float(two_year_olds[two_year_olds[x] == 1]['pred'].sum()) / n_startups[i] * 100

    global_funding_rate = float(two_year_olds['pred'].sum()) / len(two_year_olds) * 100
    print global_funding_rate

    for i, x in enumerate(two_year_olds.columns[28:106]):
        # if funding_rate[i] > global_funding_rate:
        print '%50s %6d %7.2f' % (x, n_startups[i], funding_rate[i])

    plt.style.use('fivethirtyeight')
    plt.scatter(n_startups, funding_rate, s=100, alpha=0.5)
    plt.axhline(y=global_funding_rate, color='k', linestyle='--', linewidth=0.5)
    plt.xlabel('Number of new startups in an industry')
    plt.ylabel('Predicted funding rate (%)')
    plt.title('Predicted funding rate by industry')
    plt.savefig('funding_rate_by_industry.png', dpi=400, bbox_inches='tight')

    # Port the results to MySQL
    populate_prediction_table(two_year_olds)
