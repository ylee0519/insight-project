#!/usr/bin/python

import json
import pdb
import requests
import sys
import time

import numpy as np
import pandas as pd

API_KEY = open('.crunchbase_api_key').read()

def fetch(url):
    r = requests.get(url)
    if r.status_code != 200:
        print 'Request failed. Return: '
        print
        print r.text
        sys.exit(1)

    return r.json()

def do_query(startup):
    # Get the permalink for a given startup
    url = 'https://api.crunchbase.com/v/2/organizations?domain_name=' + startup.website + '&user_key=' + API_KEY + '&page=1'
    _d = fetch(url)

    try:
        permalink = _d['metadata'][u'api_path_prefix'] + _d[u'data'][u'items'][0][u'path']
    except:
        return {u'id': startup.id}

    # Do a second query using the permalink
    url = permalink + '?user_key=' + API_KEY
    _d = fetch(url)

    try:
        d = {
            u'id': startup.id,
            u'metadata': _d[u'metadata'],
            u'description': _d[u'data'][u'properties'].get(u'description'),
            u'short_description': _d[u'data'][u'properties'].get(u'short_description'),
            u'permalink': _d[u'data'][u'properties'].get(u'permalink'),
            u'name': _d[u'data'][u'properties'].get(u'name'),
            u'homepage_url': _d[u'data'][u'properties'].get(u'homepage_url'),
            u'primary_image': _d[u'data'][u'relationships'].get(u'primary_image'),
            u'headquarters': _d[u'data'][u'relationships'].get(u'headquarters'),
            u'news': _d[u'data'][u'relationships'].get(u'news')
        }

    except KeyError as e:
        print e
        sys.exit(1)

    return d

if __name__ == '__main__':
    startups = pd.read_csv('5yo_startups_cleaned.csv')
    active_startups = startups.drop(startups.index[startups.stage == 'Exited'])

    # Only retrieve the infomation on startups that 1) have seed funding and 2) have at least 5 employees
    # Done this way because there's an API rate limit (2500 calls/day)
    active_startups = active_startups[(active_startups.seed_funding >= 1000000) & (active_startups.employee >= 5)]
    print len(active_startups), len(active_startups[active_startups.stage.isin(['A', 'B', 'C', 'Late'])])
    pdb.set_trace()

    active_startups_CB = []
    failed_attempts = 0

    for _, row in active_startups.iterrows():
        d = do_query(row)
        active_startups_CB.append(d)

        if len(d.keys()) > 1:
            print '%4d %10d %s OK' % (len(active_startups_CB), d[u'id'], time.ctime())
        else:
            failed_attempts += 1
            print '%4d %10d %s FAILED' % (len(active_startups_CB), d[u'id'], time.ctime())

        sys.stdout.flush()
        time.sleep(10)

    print 'Number of failed requests = %d' % failed_attempts

    path = 'json/5yo_active_startups_CB_1M.json'
    with open(path, 'wb') as f:
        json.dump(active_startups_CB, f, -1)
