"""No GPU required: admission, shared lock, and scoped emergency stop."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / 'llama-swap/scripts/guard-large-model.sh'

class GuardTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / 'scripts').mkdir()
        (self.root / 'bin').mkdir()
        self.script = self.root / 'scripts/guard.sh'
        shutil.copy2(SCRIPT, self.script)
        self.env = dict(os.environ, MOCK_ROOT=str(self.root), PATH=str(self.root/'bin')+':'+os.environ['PATH'])
        self.mock('sleep', '#!/bin/sh\n/bin/sleep 0.05\n')
        self.mock('awk', '''#!/usr/bin/env python3
import os,pathlib
p=pathlib.Path(os.environ['MOCK_ROOT'])/'reads'
n=int(p.read_text()) if p.exists() else 0
p.write_text(str(n+1))
print(120*1048576 if os.environ.get('CASE')=='pass' or (os.environ.get('CASE')=='watchdog' and n<3) else 5*1048576)
''')
        self.mock('docker', '''#!/usr/bin/env python3
import os,pathlib,sys,time,signal
p=pathlib.Path(os.environ['MOCK_ROOT']);args=sys.argv[1:]
with (p/'docker.log').open('a') as f:f.write(' '.join(args)+'\\n')
if args[0]=='ps':print('router-id')
if args[0]=='run':
 pathlib.Path(args[args.index('--cidfile')+1]).write_text('a'*64)
 (p/'pid').write_text(str(os.getpid()))
 time.sleep(5)
if args[0]=='kill':
 assert args[1]=='a'*64
 try:os.kill(int((p/'pid').read_text()),signal.SIGTERM)
 except ProcessLookupError:pass
''')
    def mock(self,name,text):
        p=self.root/'bin'/name;p.write_text(text);p.chmod(0o755)
    def run_guard(self,case,*args):
        return subprocess.run(['bash',str(self.script),'test-model','104',*args],env=dict(self.env,CASE=case),capture_output=True,text=True,timeout=10)
    def tearDown(self):self.tmp.cleanup()
    def test_low_memory_never_calls_docker(self):
        r=self.run_guard('low','image')
        self.assertEqual(r.returncode,75)
        self.assertFalse((self.root/'docker.log').exists())
    def test_check_only_does_not_launch(self):
        self.assertEqual(self.run_guard('pass','--check-only').returncode,0)
        self.assertFalse((self.root/'docker.log').exists())
    def test_shared_lock_blocks_launch(self):
        import fcntl
        (self.root/'runtime').mkdir()
        with (self.root/'runtime/large-model.lock').open('w') as f:
            fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
            self.assertEqual(self.run_guard('pass','image').returncode,75)
        self.assertFalse((self.root/'docker.log').exists())
    def test_watchdog_kills_only_owned_container(self):
        r=self.run_guard('watchdog','image')
        self.assertNotEqual(r.returncode,0)
        self.assertIn('EMERGENCY STOP',r.stderr)
        log=(self.root/'docker.log').read_text()
        self.assertIn('kill '+'a'*64,log)
        self.assertNotIn('kill router-id',log)
        self.assertFalse(list((self.root/'runtime').glob('container.*')))

if __name__=='__main__':unittest.main()
