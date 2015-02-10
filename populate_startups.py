#!/usr/bin/python

import json
import numpy as np
import pandas as pd
import pdb

from datastore import local

startups = pd.read_csv('5yo_startups_cleaned.csv')
active_startups_CB = json.load(open('json/two_year_olds_CB.json'))
print len(startups), len(active_startups_CB)

db = local()
q = '''DROP TABLE IF EXISTS startups;'''
db.cur.execute(q)

q = '''CREATE TABLE startups (Id INT, Name VARCHAR(255), Homepage VARCHAR(255), Description TEXT, ShortIntro TEXT, Stage VARCHAR(12), Location VARCHAR(255), Mobile TINYINT(1), Ecommerce TINYINT(1), Social TINYINT(1), Enterprise TINYINT(1), Marketing TINYINT(1), Analytics TINYINT(1), Healthcare TINYINT(1), Media TINYINT(1), Advertising TINYINT(1), Education TINYINT(1));'''
db.cur.execute(q)

for x in active_startups_CB:
    if len(x.keys()) > 1:
        index = x['id']
        name = x['name']
        homepage = x['homepage_url']
        description = x['description']
        short_intro = x['short_description']

        i = startups[startups.id == index].index[0]
        stage = startups.at[i, 'stage']
        loc = startups.at[i, 'location']
        mobile = int(int(startups.at[i, 'mobile']) == 1)
        ecommerce = int(int(startups.at[i, 'e-commerce']) == 1)
        social = int(int(startups.at[i, 'social networking']) == 1)
        enterprise = int(int(startups.at[i, 'enterprise software']) == 1)
        marketing = int(int(startups.at[i, 'marketing']) == 1)
        analytics = int(int(startups.at[i, 'analytics']) == 1)
        healthcare = int(int(startups.at[i, 'healthcare']) == 1)
        media = int(int(startups.at[i, 'media']) == 1)
        advertising = int(int(startups.at[i, 'advertising']) == 1)
        education = int(int(startups.at[i, 'education']) == 1)

        q = '''INSERT INTO startups VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);'''
        db.cur.execute(q, (
            index, 
            name,
            homepage,
            description,
            short_intro,
            stage,
            loc,
            mobile,
            ecommerce,
            social,
            enterprise,
            marketing,
            analytics,
            healthcare,
            media,
            advertising,
            education))

db.con.commit()
db.close()
