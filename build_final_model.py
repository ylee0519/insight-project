#!/usr/bin/python

import numpy as np
import pandas as pd
import pdb

from datastore import local

from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import RandomForestClassifier

def run_model(X, y):
    clf = RandomForestClassifier(n_jobs=2)
    clf = clf.fit(X, y)
    p = 100*clf.predict_proba(X)

    return p[:, 1].astype(int)

def populate_scores_table(active_startups):
    db = local()
    q = '''DROP TABLE IF EXISTS scores;'''
    db.cur.execute(q)

    q = '''CREATE TABLE scores (Id INT, Score INT);'''
    db.cur.execute(q)

    for i, row in active_startups.iterrows():
        q = '''INSERT INTO scores VALUES (%s, %s);'''
        db.cur.execute(q, (row.id, row.score))

    db.con.commit()
    db.close()

if __name__ == '__main__':
    # Pre-processing of data
    startups = pd.read_csv('5yo_startups_cleaned.csv')
    active_startups = startups.drop(startups.index[startups.stage == 'Exited'])
    y = active_startups.stage.isin(['A', 'B', 'C', 'Late']).astype(int)
    X = active_startups.drop([
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
    active_startups[u'score'] = run_model(X, y)
    
    # Port the results to MySQL
    populate_scores_table(active_startups)
