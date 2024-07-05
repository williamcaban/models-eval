
curl -k http://vllm-vllm.apps.fennec.shift.zone/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
     "model":"instructlab/granite-7b-lab",
     "messages": [{"role": "user", "content": "What is Red Hat?"}],
     "temperature": 0.7
   }'

