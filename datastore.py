#!/usr/bin/python

import MySQLdb as mdb
import os
import sys

PASSWORD = open('.mysql_root_passwd').read()

class local(object):
    def __init__(self, verbose=False):
        try:
            self.con = mdb.connect(host='localhost', user='root', passwd=PASSWORD, db='insight_proj', charset='utf8', use_unicode=True)
            self.cur = self.con.cursor()
            if verbose:
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
