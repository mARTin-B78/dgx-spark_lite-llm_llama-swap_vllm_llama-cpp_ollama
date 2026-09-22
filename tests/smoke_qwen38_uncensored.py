"""Live admission, coding, tools, 60K recall and streamed throughput checks."""
import ast,json,time,threading
from pathlib import Path
import requests
OUT=Path('test-results/qwen38-uncensored-20260917');OUT.mkdir(parents=True,exist_ok=True)
MODEL='Qwen3.8-27B-Uncensored-NVFP4-DFlash2'
BASE='http://127.0.0.1:28080/v1/chat/completions'
stop=threading.Event()
def monitor():
 with (OUT/'memory.csv').open('w') as f:
  f.write('unix,available_kib,swap_used_kib\n')
  while not stop.is_set():
   m={s.split(':')[0]:int(s.split()[1]) for s in Path('/proc/meminfo').read_text().splitlines()}
   f.write(f"{time.time()},{m['MemAvailable']},{m['SwapTotal']-m['SwapFree']}\n");f.flush();stop.wait(1)
threading.Thread(target=monitor,daemon=True).start()
def run(name,body):
 payload={'model':MODEL,'temperature':0,'max_tokens':512,'chat_template_kwargs':{'enable_thinking':False},**body}
 t=time.monotonic();r=requests.post(BASE,json=payload,timeout=1200)
 data=r.json();record={'http':r.status_code,'seconds':round(time.monotonic()-t,3),'response':data}
 (OUT/(name+'.json')).write_text(json.dumps(record,indent=2)+'\n');r.raise_for_status()
 print(name,record,flush=True);return data['choices'][0]['message']
try:
 msg=run('coding',{'messages':[{'role':'user','content':'Write a Python function first_duplicate(xs) that returns the first value encountered a second time, or None. Use a set. Return only the function.'}]})
 code=msg['content'];code=code.split('```python')[-1].split('```')[0].strip() if '```python' in code else code.strip('`\n')
 tree=ast.parse(code)
 assert all(not isinstance(n,(ast.Import,ast.ImportFrom)) for n in ast.walk(tree))
 for n in ast.walk(tree):
  if isinstance(n,ast.Call):assert isinstance(n.func,ast.Name) and n.func.id=='set' or isinstance(n.func,ast.Attribute) and n.func.attr=='add'
 assert len(tree.body)==1 and isinstance(tree.body[0],ast.FunctionDef)
 ns={'__builtins__':{'set':set}};exec(compile(tree,'model_function','exec'),ns)
 for xs,expected in [([],None),([1,2,3],None),([1,2,1],1),(['a','b','b'],'b'),([-1,0,-1],-1)]:assert ns['first_duplicate'](xs)==expected
 msg=run('tool',{'messages':[{'role':'user','content':'Call get_weather for Berlin.'}],'tools':[{'type':'function','function':{'name':'get_weather','description':'Get weather','parameters':{'type':'object','properties':{'city':{'type':'string'}},'required':['city']}}}],'tool_choice':'auto'})
 call=msg['tool_calls'][0]['function'];assert call['name']=='get_weather' and json.loads(call['arguments'])['city']=='Berlin'
 text='alpha beta gamma delta. '*6000+'\nThe secret verification code is ORCHID-7291.\n'+'alpha beta gamma delta. '*6000
 msg=run('long-context',{'messages':[{'role':'user','content':text+'\nReturn only the secret verification code from the document.'}]})
 assert 'ORCHID-7291' in msg['content']
 t=time.monotonic();first=None;last=None;content='';usage={}
 with requests.post(BASE,json={'model':MODEL,'messages':[{'role':'user','content':'Write a detailed Python LRU cache implementation with a doubly linked list and a dictionary. Include type annotations and explain invariants.'}],'max_tokens':512,'temperature':0,'chat_template_kwargs':{'enable_thinking':False},'stream':True,'stream_options':{'include_usage':True}},stream=True,timeout=300) as r:
  r.raise_for_status()
  for line in r.iter_lines():
   if not line or not line.startswith(b'data: '):continue
   raw=line[6:]
   if raw==b'[DONE]':break
   data=json.loads(raw);usage=data.get('usage') or usage
   for choice in data.get('choices',[]):
    c=choice.get('delta',{}).get('content') or ''
    if c:
     now=time.monotonic();first=first or now;last=now;content+=c
 result={'ttft_seconds':first-t if first else None,'elapsed_seconds':time.monotonic()-t,'usage':usage,'content':content,'approx_decode_tokens_per_second':(usage.get('completion_tokens',0)-1)/(last-first) if first and last>first else None}
 (OUT/'throughput.json').write_text(json.dumps(result,indent=2)+'\n');print('throughput',result,flush=True)
 assert content and usage.get('completion_tokens',0)>0
 print('ALL CHECKS PASSED',flush=True)
finally:stop.set()
