#!/usr/bin/python

import json
import os

from datastore import local
from flask import *

app = Flask(__name__)

PER_PAGE = 20

@app.route('/')
def hello():
    return render_template('hello.html')

@app.route('/<industry>')
def result(industry):
    db = local()
    q = '''SELECT startups.Name, 
    startups.Homepage, 
    startups.ShortIntro, 
    startups.Stage,
    prediction.Score
    FROM startups INNER JOIN prediction ON startups.Id = prediction.Id 
    WHERE startups.%s = 1;
    ''' % industry.capitalize()

    db.cur.execute(q)
    rows = db.cur.fetchall()
    count = len(rows)
    startups = []

    for _, row in enumerate(rows):
        if row[3] == 'Pre Series A':
            startup = {'name': row[0],
                'homepage': row[1],
                'short_intro': row[2],
                'stage': 'Pre A',
                'score': row[4]
                }
            startups.append(startup)

    db.close()

    return render_template('result.html',
        industry=industry,
        startups=startups
        )

@app.errorhandler(404)
def page_not_found(e):
    """Return a custom 404 error."""
    return 'Sorry, Nothing at this URL.', 404

@app.errorhandler(500)
def unexpected_error(e):
    """Return a custom 500 error."""
    return 'Sorry, unexpected error: {}'.format(e), 500

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)
