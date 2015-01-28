#!/usr/bin/python

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
    active_startups['age'] = pd.to_datetime(datetime(2015, 1, 1)) - pd.to_datetime(active_startups.est_founding_date)
    
    two_year_olds = active_startups[active_startups.age <= timedelta(365*2)].copy()
    toddlers = active_startups[active_startups.age > timedelta(365*2)].copy()

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

    # Train and test the model with all the data
    clf = RandomForestClassifier(n_jobs=2)
    clf = clf.fit(X_train, y_train)

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

    print clf.score(X_test, y_test)

    pred = 100*clf.predict_proba(X_test).astype(float)

    two_year_olds[u'score'] = pd.Series(pred[:, 1].astype(int), index=two_year_olds.index)
    
    # Port the results to MySQL
    populate_prediction_table(two_year_olds)
