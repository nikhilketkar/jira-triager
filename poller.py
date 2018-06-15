import time,datetime
import sys
from jira import JIRA

fields="key,Affected Component,summary"
max_results=10

old_new = {"Batch Processing":"Batch Processing",
"Data Quality":"Data Quality",
"I don't know":"None",
"Infrastructure":"Infrastructure",
"Ingestion":"Ingesiton",
"Krios":"Infrastructure",
"Machine Learning":"Adhoc Query",
"Pre Processing":"Batch Processing",
"Reporting & Dashboards":"Report & Dashboard",
"Seraph (Self-Serve UI)":"Access",
"Stream Processing":"Stream processing"}

def get_issues(user, password, jql, fields, max_results):
    options = {'server': 'https://jira.fkinternal.com'}
    jira = JIRA(options = options, basic_auth = (user, password))
    issues = jira.search_issues(jql_str=jql, maxResults=max_results, fields=fields)
    return issues

def write_data(output_path, issues):
    with open(output_path, 'w') as output_file:
        for i in issues:
            output_file.write("\t".join([i.key, i.fields.customfield_17605.value, i.fields.summary]) + "\n")
 

def startPoller(sleepTime, output_path, user, password):
	while True:
		t = str(datetime.datetime.now().strftime("%Y/%m/%d %H:%M"))
		time.sleep(sleepTime)
		jql = constructJQL(t)
		issues = get_issues(user,password,jql,fields,max_results)	
		write_data(output_path, issues)

def constructJQL(t):
	jql = "project = \"FDPON\" AND created > \""+t+"\" ORDER BY created DESC"
	return jql


def updateJira(id,value,jira):
	issue = jira.issue(id)
	issue.update(fields={"customfield_17605":{"value":value}})




startPoller(int(sys.argv[1]),sys.argv[2],sys.argv[3],sys.argv[4])
