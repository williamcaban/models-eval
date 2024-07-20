# Create SA with OLS role and generate TOKEN for app

OLS_API="https://"$(oc get route api -n openshift-lightspeed -o jsonpath='{.spec.host}')"/v1/query"
OLS_ROLE="lightspeed-operator-query-access"
SA_QUERY="sa-ols-query"

oc create sa $SA_QUERY
oc create clusterrolebinding $SA_QUERY --clusterrole=$OLS_ROLE --serviceaccount=openshift-lightspeed:$SA_QUERY

# Option 1
export TOKEN=$(oc create token -n openshift-lightspeed $SA_QUERY --duration=0s)

# Option 2
oc apply -f - <<EOF
apiVersion: v1
kind: Secret
metadata:
  name: $SA_QUERY-secret
  namespace: openshift-lightspeed
  annotations:
    kubernetes.io/service-account.name: $SA_QUERY
type: kubernetes.io/service-account-token
EOF

export TOKEN2=$(oc get secret $SA_QUERY-secret -o jsonpath='{.data.token}')

echo "OLS_API_TOKEN  = "$TOKEN  >  OLS_API_TOKEN.txt
echo "OLS_API_TOKEN2 = "$TOKEN2 >> OLS_API_TOKEN.txt
