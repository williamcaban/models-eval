#!/usr/bin/env python3.11
#
import requests
import urllib3
urllib3.disable_warnings()

class olsClient():
    def __init__(self, api_url, token, llm_provider, model) -> None:
        """
        api_url : URL to OpenShift Lightspeed query endpoint
        token   : ServiceAccount token for the API
        """
        self.OLS_ENDPOINT = api_url
        self.TOKEN=token
        self.LLM_PROVIDER= llm_provider
        self.MODEL = model

    def query(self, user_query):
        # Define header for making API calls that will hold authentication data
        headersAPI = {
            'accept': 'application/json',
            'Content-Type': 'application/json',
            'User-Agent': 'Mozilla/5.0 (compatible; OLS Eval/1.0; en-us)',
            'Authorization': f'Bearer {self.TOKEN}',
        }

        response = requests.post(
            url=self.OLS_ENDPOINT,
            headers=headersAPI,
            json={"query": user_query, "provider": self.LLM_PROVIDER,
                  "model": self.MODEL},
            verify=False)
        
        if response.ok:
            return response.json()['response']
        else:
            print(f"Something went wrong {response.status_code} ==> {response.json()}")
            return None