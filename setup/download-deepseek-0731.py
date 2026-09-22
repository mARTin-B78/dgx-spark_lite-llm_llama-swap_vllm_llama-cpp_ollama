import concurrent.futures, hashlib, json, os, pathlib, requests, time
repo='antirez/deepseek-v4-gguf'
name='DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf'
root=pathlib.Path('/home/sparky/LLMs/ollama/DeepSeek/0731')
revision='f71f23d552d664e523b422157b2befbf74040380'
info=requests.get(f'https://huggingface.co/api/models/{repo}/revision/{revision}',params={'blobs':'true'},timeout=30);info.raise_for_status();info=info.json()
f=next(x for x in info['siblings'] if x['rfilename']==name)
size=f['size'];sha=f['lfs']['sha256'];rev=info['sha']
url=f'https://huggingface.co/{repo}/resolve/{rev}/{name}'
target=root/name;part=root/(name+'.part');state=root/(name+'.parts.json')
if target.exists():
 receipt=root/'download-receipt.json'
 if receipt.exists() and json.loads(receipt.read_text()).get('sha256')==sha and target.stat().st_size==size:
  print('Already downloaded and verified: '+str(target));raise SystemExit(0)
 raise SystemExit('Existing target has no matching receipt; verify it before replacing')
chunk=64*1024*1024;done=set(json.loads(state.read_text()) if state.exists() else [])
fd=os.open(part,os.O_RDWR|os.O_CREAT,0o644);os.ftruncate(fd,size)
def fetch(i):
 start=i*chunk;end=min(size,start+chunk)-1
 for attempt in range(8):
  try:
   with requests.get(url+f'?part={i}',headers={'Range':f'bytes={start}-{end}'},stream=True,timeout=(20,60)) as r:
    if r.status_code!=206 or r.headers.get('Content-Range')!=f'bytes {start}-{end}/{size}':raise RuntimeError('Invalid range response')
    offset=start
    for data in r.iter_content(1024*1024):
     if offset+len(data)>end+1:raise RuntimeError('Oversized range')
     view=memoryview(data)
     while view:
      n=os.pwrite(fd,view,offset);offset+=n;view=view[n:]
    if offset!=end+1:raise RuntimeError('Incomplete range')
   return i
  except Exception as e:
   if attempt==7:raise RuntimeError(f'Chunk {i} failed: {type(e).__name__}') from None
   time.sleep(min(15,attempt+1))
started=time.monotonic();last=started
print(f'Downloading {size/2**30:.2f} GiB; revision {rev}; SHA256 {sha}',flush=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:
 for future in concurrent.futures.as_completed([pool.submit(fetch,i) for i in range((size+chunk-1)//chunk) if i not in done]):
  done.add(future.result());state.write_text(json.dumps(sorted(done)))
  now=time.monotonic()
  if now-last>20:
   print(f'{len(done)*chunk/2**30:.2f}/{size/2**30:.2f} GiB complete; elapsed {now-started:.0f}s',flush=True);last=now
os.fsync(fd);os.close(fd)
print('Download finished; verifying SHA256',flush=True)
h=hashlib.sha256()
with part.open('rb') as inp:
 for data in iter(lambda:inp.read(8*1024*1024),b''):h.update(data)
if h.hexdigest()!=sha:raise SystemExit('SHA256 MISMATCH; incomplete file retained')
part.rename(target)
(root/'download-receipt.json').write_text(json.dumps({'repo':repo,'revision':rev,'file':name,'bytes':size,'sha256':sha,'verified':True},indent=2)+'\n')
state.unlink()
print('VERIFIED '+str(target),flush=True)
