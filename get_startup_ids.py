#!/usr/bin/python

import cPickle
import json
import requests
import sys
import time

API_KEY = open('.api_key').read()

# BASE_URL = 'https://mattermark.com/app/v0/companies/?key=' + API_KEY + '&country=USA&est_founding_date=within+5+years&stage=Pre+Series+A|a|b|c|Late|Exited'
# SAVE_TO = '5yo_startup_ids.pkl'

BASE_URL = 'https://mattermark.com/app/v0/companies/?key=' + API_KEY + '&industries=Mobile&country=USA&est_founding_date=within+10+years&stage=Pre+Series+A|a|b|c|Late|Exited'
SAVE_TO = 'versionone/10yo_mobile_startup_ids.pkl'

def get_json(url):
    r = requests.get(url)
    if r.status_code != 200:
        print 'Request failed. Return: ',
        print r.text
        sys.exit(1)
    
    d = r.json()

    return d

def do_query(offset):
    url = BASE_URL + '&offset=%d' % (offset + 1)
    d = get_json(url)

    g = []
    for item in d[u"companies"]:
        g.append(item[u"id"])

    return g

if __name__ == '__main__':
    d = get_json(BASE_URL)
    total_companies = int(d[u'total_companies'])
    print total_companies
    raw_input('--- PRESS ANY KEY ---')

    company_ids = []
    step = 50
    for i in xrange(total_companies/step + 1):
        g = do_query(i*step)
        company_ids.extend(g)
        print '%10d %s' % (len(company_ids), time.ctime())

    print len(company_ids)
    
    with open(SAVE_TO, 'wb') as f:
        cPickle.dump(company_ids, f, -1)
