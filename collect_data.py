from jira import JIRA
from dateutil.parser import parse
import pandas
from StringIO import StringIO

fields="key,FDP Component,summary"

def get_issues(user, password, fields, max_results):
    options = {'server': 'https://jira.fkinternal.com'}
    jira = JIRA(options = options, basic_auth = (user, password))
    issues = jira.search_issues("project = FDPON", maxResults=max_results, fields=fields)
    return issues

def write_data(output_path, issues):
    with open(output_path, 'w') as output_file:
        for i in issues:
            output_file.write("\t".join([i.key, i.fields.customfield_13501.value, i.fields.summary.encode('ascii','ignore')]) + "\n")
                              



