import json
import os

from datastore import local
from flask import *

app = Flask(__name__)

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
    scores.Score
    FROM startups INNER JOIN scores ON startups.Id = scores.Id 
    WHERE startups.%s = 1;
    ''' % industry.capitalize()

    top_startups = []

    db.cur.execute(q)
    rows = db.cur.fetchall()

    for i, row in enumerate(rows[:50]):
        startup = {'name': row[0],
            'homepage': row[1],
            'short_intro': row[2],
            'stage': row[3] if row[3] != 'Pre Series A' else 'Pre A',
            'score': row[4]
            }

        top_startups.append(startup)

    db.close()

    return render_template('result.html', startups=top_startups)

@app.errorhandler(404)
def page_not_found(e):
    """Return a custom 404 error."""
    return 'Sorry, Nothing at this URL.', 404

@app.errorhandler(500)
def page_not_found(e):
    """Return a custom 500 error."""
    return 'Sorry, unexpected error: {}'.format(e), 500

if __name__ == '__main__':
    app.run()
