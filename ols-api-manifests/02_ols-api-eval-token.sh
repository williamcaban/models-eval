# Create SA with OLS role and generate TOKEN for app

OLS_API="https://"$(oc get route api -n openshift-lightspeed -o jsonpath='{.spec.host}')"/v1/query"
OLS_ROLE="lightspeed-operator-query-access"
SA_QUERY="sa-ols-query"

oc create sa $SA_QUERY
oc create clusterrolebinding $SA_QUERY --clusterrole=$OLS_ROLE --serviceaccount=openshift-lightspeed:$SA_QUERY

export TOKEN=$(oc create token -n openshift-lightspeed $SA_QUERY --duration=0s)

echo "OLS_API_TOKEN="$TOKEN > OLS_API_TOKEN.txt
