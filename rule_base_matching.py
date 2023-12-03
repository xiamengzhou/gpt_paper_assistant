import dataclasses
from datetime import datetime

import pytz

from arxiv_scraper import get_papers_from_arxiv_api
from parse_json_to_md import render_md_string


def convert_date(time_string="2023-12-03 12:00:00"):
    # Parse the string into a datetime object
    # Assuming the format is Year-Month-Day Hour:Minute:Second
    dt = datetime.strptime(time_string, "%Y-%m-%d %H:%M:%S")

    # Localize the datetime object to EST
    est = pytz.timezone("US/Eastern")
    dt_est = est.localize(dt)

    # Convert to UTC
    dt_utc = dt_est.astimezone(pytz.utc)
    return dt_utc

def match_keyword(paper, keyword_list):
    def match(keyword):
        return paper.abstract.lower().find(keyword) != -1 or paper.title.lower().find(keyword) != -1

    for keyword in keyword_list:
        if match(keyword):
            return True
        
    return False 

areas = ["cs.CL", "cs.AI", "cs.LG"]

area = "cs.CL"
keyword_list = ["instruction tuning", "feedback", "preference"]

last_date = convert_date("2023-12-03 12:00:00")
delta=3

papers = get_papers_from_arxiv_api(area, timestamp=last_date, delta=delta)
selected_papers = dict()
for paper in papers:
    match = match_keyword(paper, keyword_list)
    if match:
        selected_papers[paper.arxiv_id] = dataclasses.asdict(paper)

with open("database/index.md", "w") as f:
    f.write(render_md_string(selected_papers))