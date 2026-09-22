"""Small live checks; not a coding benchmark. Run one already-admitted model."""
import argparse,json,time
from pathlib import Path
import requests
p=argparse.ArgumentParser();p.add_argument('model');p.add_argument('--long',action='store_true');a=p.parse_args()
base='http://127.0.0.1:28080/v1/chat/completions'
out=Path('test-results/ssd-ds4-20260916');out.mkdir(parents=True,exist_ok=True)
common={'model':a.model,'temperature':0,'max_tokens':512}
if a.model.startswith('Qwen'):common['chat_template_kwargs']={'enable_thinking':False}
else:common['thinking']={'type':'disabled'}
cases=[('coding',{'messages':[{'role':'user','content':'Write a Python function first_duplicate(xs) that returns the first value encountered a second time, or None. Use a set. Return only the function.'}]}),('tool',{'messages':[{'role':'user','content':'Call get_weather for Berlin.'}],'tools':[{'type':'function','function':{'name':'get_weather','description':'Get weather','parameters':{'type':'object','properties':{'city':{'type':'string'}},'required':['city']}}}],'tool_choice':'auto'})]
if a.long:
 text=('alpha beta gamma delta. '*8000)+'\nThe secret verification code is ORCHID-7291.\n'+('alpha beta gamma delta. '*8000)
 cases.append(('long-context',{'messages':[{'role':'user','content':text+'\nReturn only the secret verification code from the document.'}]}))
for name,case in cases:
 start=time.monotonic();r=requests.post(base,json={**common,**case},timeout=900)
 result={'http':r.status_code,'seconds':round(time.monotonic()-start,2),'response':r.json()}
 (out/(a.model+'-'+name+'.json')).write_text(json.dumps(result,indent=2)+'\n')
 print(name,result,flush=True)
 r.raise_for_status()
 msg=r.json()['choices'][0]['message']
 if name=='tool':
  assert msg.get('tool_calls'), 'No tool call'
  call=msg['tool_calls'][0]['function']
  assert call['name']=='get_weather' and json.loads(call['arguments'])['city']=='Berlin'
 if name=='long-context':assert 'ORCHID-7291' in (msg.get('content') or ''), 'Recall failed'
 if name=='coding':assert 'def first_duplicate' in (msg.get('content') or ''), 'Missing function'
