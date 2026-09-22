"""Pinned, resumable range download; verify every weight shard before use."""
import concurrent.futures, hashlib, json, os, pathlib, requests, time
repo='Vtuber-plan/Qwen3.8-27B-Uncensored-NVFP4'
rev='b0f5a99384f9db15e22d028a98d8302284e3748d'
root=pathlib.Path('/home/sparky/LLMs/qwen38')/repo
root.mkdir(parents=True,exist_ok=True)
r=requests.get(f'https://huggingface.co/api/models/{repo}/revision/{rev}',params={'blobs':'true'},timeout=30);r.raise_for_status()
files=[f for f in r.json()['siblings'] if f['rfilename'].endswith(('.safetensors','.json','.jinja','.txt','.md'))]
chunk=32*1024**2
states={};jobs=[];handles={}
for f in files:
 name=f['rfilename'];p=root/name;p.parent.mkdir(parents=True,exist_ok=True)
 if 'lfs' not in f:
  r=requests.get(f'https://huggingface.co/{repo}/resolve/{rev}/{name}',timeout=60);r.raise_for_status();p.write_bytes(r.content);continue
 if p.exists():
  with p.open('rb') as inp: digest=hashlib.file_digest(inp,'sha256').hexdigest()
  if digest==f['lfs']['sha256']:continue
  raise RuntimeError('Existing file checksum mismatch: '+name)
 state=root/(name+'.ranges.json');done=set(json.loads(state.read_text()) if state.exists() else [])
 states[name]=(state,done);fd=os.open(str(p)+'.part',os.O_CREAT|os.O_RDWR,0o644);os.ftruncate(fd,f['size']);handles[name]=fd
 jobs.extend((f,i) for i in range((f['size']+chunk-1)//chunk) if i not in done)
def fetch(job):
 f,i=job;name=f['rfilename'];size=f['size'];start=i*chunk;end=min(size,start+chunk)-1
 for attempt in range(8):
  try:
   with requests.get(f'https://huggingface.co/{repo}/resolve/{rev}/{name}?range={i}',headers={'Range':f'bytes={start}-{end}'},stream=True,timeout=(20,60)) as r:
    if r.status_code!=206 or r.headers.get('Content-Range')!=f'bytes {start}-{end}/{size}':raise RuntimeError('Invalid range')
    offset=start
    for data in r.iter_content(1024**2):
     if offset+len(data)>end+1:raise RuntimeError('Oversized range')
     view=memoryview(data)
     while view:
      n=os.pwrite(handles[name],view,offset);offset+=n;view=view[n:]
    if offset!=end+1:raise RuntimeError('Incomplete range')
   return name,i
  except Exception:
   if attempt==7:raise
   time.sleep(min(15,attempt+1))
jobs.sort(key=lambda job: (job[1], job[0]["rfilename"]))
last=time.monotonic();completed=0
print('Downloading',len(jobs),'ranges',flush=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=64) as pool:
 for fut in concurrent.futures.as_completed([pool.submit(fetch,j) for j in jobs]):
  name,i=fut.result();state,done=states[name];done.add(i)
  tmp=state.with_suffix('.tmp');tmp.write_text(json.dumps(sorted(done)));tmp.replace(state);completed+=1
  if time.monotonic()-last>20:
   print(f'{completed}/{len(jobs)} ranges completed',flush=True);last=time.monotonic()
for fd in handles.values():os.fsync(fd);os.close(fd)
receipt=[]
for f in files:
 if 'lfs' not in f:continue
 p=root/f['rfilename'];part=pathlib.Path(str(p)+'.part');src=p if p.exists() else part
 with src.open('rb') as inp:digest=hashlib.file_digest(inp,'sha256').hexdigest()
 assert digest==f['lfs']['sha256'],f['rfilename']
 if src==part:part.replace(p)
 state=root/(f['rfilename']+'.ranges.json');state.unlink(missing_ok=True)
 receipt.append({'file':f['rfilename'],'sha256':digest,'bytes':p.stat().st_size})
(root/'download-receipt.json').write_text(json.dumps({'repo':repo,'revision':rev,'verified':True,'files':receipt},indent=2)+'\n')
print('ALL SHARDS VERIFIED',flush=True)
