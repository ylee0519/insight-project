#!/usr/bin/python

import json
import os
import requests
import sys
import threading
import time

from utils import unpickle_file_at_path

API_KEY = open('.api_key').read()
BASE_URL = 'https://mattermark.com/app/v0/companies'

# Configs for insight project
# PATH_TO_IDS = '5yo_startup_ids.pkl'
# PATH_TO_COMPANIES = '5yo_startups.json'

# Configs for versionone
PATH_TO_IDS = 'versionone/10yo_mobile_startup_ids.pkl'
PATH_TO_COMPANIES = 'versionone/10yo_mobile_startups.json'

class QueryJob(object):
    def __init__(self, company_ids, num_threads=20):
        self.company_ids = company_ids
        self.num_threads = num_threads
        self.lock = threading.Lock()
        self.thread_grp = []
        self.companies = []

    def do_query(self, company_id):
        opts = {
            "key": API_KEY
        }

        r = requests.get(os.path.join(BASE_URL, company_id), params=opts)
        if r.status_code != 200:
            print 'Request failed. Return: ',
            print r.text
            sys.exit(1)

        d = r.json()
        d[u'id'] = company_id

        # # Twitter: Total followers to date
        # if len(d[u'twitter_follower_count']) > 0:
        #     d[u'twitter_followers_total'] = int(d[u'twitter_follower_count'][0][u'score'])
        # else:
        #     d[u'twitter_followers_total'] = 0

        self.lock.acquire()
        self.companies.append(d)
        self.lock.release()

    def run(self):
        for i in xrange(len(self.company_ids)):
            company_id = company_ids[i]
            t = threading.Thread(target=self.do_query, args=(company_id, ))
            t.start()
            self.thread_grp.append(t)

            if len(self.thread_grp) >= self.num_threads:
                while len(self.thread_grp):
                    _t = self.thread_grp.pop()
                    _t.join()

                print '%5d %s' % (len(self.companies), time.ctime())

        while len(self.thread_grp):
            _t = self.thread_grp.pop()
            _t.join()

        print '%5d %s' % (len(self.companies), time.ctime())

if __name__ == '__main__':
    try:
        company_ids = unpickle_file_at_path(PATH_TO_IDS)
    except IOError:
        print 'Nothing at %s' % PATH_TO_IDS
        sys.exit(1)

    print 'Total number of startups = %d' % len(company_ids)
    raw_input('--- PRESS ANY KEY ---')

    Q = QueryJob(company_ids)
    Q.run()
    
    # Save to json
    with open(PATH_TO_COMPANIES, 'wb') as f:
        json.dump(Q.companies, f)
