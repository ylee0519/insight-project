#!/usr/bin/python

import cPickle
import os
import sys
import time

__all__ = ['unpickle_file_at_path']

def unpickle_file_at_path(path):
    try:
        with open(path, 'rb') as f:
            sys.stdout.flush()
            t_start = time.time()
            content = cPickle.load(f)

    except IOError:
        print 'No file at %s' % path
        raise IOError

    return content
