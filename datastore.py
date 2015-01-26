#!/usr/bin/python

import MySQLdb as mdb
import os
import sys

class local(object):
    def __init__(self):
        try:
            self.con = mdb.connect(host='localhost', user='root', db='insight_proj', charset='utf8', use_unicode=True)
            self.cur = self.con.cursor()
            print "Connection is successful"

        except mdb.Error, e:
            print "Error %d: %s" % (e.args[0], e.args[1])
            sys.exit(1)

    def close(self):
        if self.con:
            self.con.close()

if __name__ == "__main__":
    # Test connection
    db = local()
