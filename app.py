import json
import os

from datastore import local
from flask import *

from flask_utils import Pagination

app = Flask(__name__)

PER_PAGE = 20

def get_statups_for_page(industry, page, per_page):
    db = local()
    q = '''SELECT startups.Name, 
    startups.Homepage, 
    startups.ShortIntro, 
    startups.Stage,
    scores.Score
    FROM startups INNER JOIN scores ON startups.Id = scores.Id 
    WHERE startups.%s = 1;
    ''' % industry.capitalize()

    db.cur.execute(q)
    rows = db.cur.fetchall()
    startups = []

    for _, row in enumerate(rows[(page-1)*per_page:page*per_page]):
        startup = {'name': row[0],
            'homepage': row[1],
            'short_intro': row[2],
            'stage': row[3] if row[3] != 'Pre Series A' else 'Pre A',
            'score': row[4]
            }

        startups.append(startup)

    db.close()

    return startups, len(rows)

@app.route('/')
def hello():
    return render_template('hello.html')

@app.route('/<industry>', defaults={'page': 1})
@app.route('/<industry>/page/<int:page>')
def result(industry, page):
    startups, count = get_statups_for_page(industry, page, PER_PAGE)
    if not startups and page != 1:
        abort(404)
    pagination = Pagination(page, PER_PAGE, count)

    return render_template('result.html',
        pagination=pagination,
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
    app.run(debug=True)
