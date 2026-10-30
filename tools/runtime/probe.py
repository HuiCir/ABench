"""Opt-in Docker assumptions probe (P0), leaving only a JSON report."""
import argparse
import json
from pathlib import Path
import statistics
import subprocess
import tempfile
import time
import uuid
from containers import Docker

def probe(image):
    d=Docker();run='probe-'+uuid.uuid4().hex[:12];info=d.info();result={'docker':{k:v for k,v in info.items() if k!='raw'},'image':image}
    volume=d.volume('abench-'+run,run)
    server=None
    try:
        d.run(['run','--rm','--label',f'abench.run={run}','--network','none','--user','0:0','--mount',f'type=volume,src={volume},dst=/socket',image,'chown','1000:1000','/socket'])
        program="import socket,os; s=socket.socket(socket.AF_UNIX);s.bind('/socket/proxy.sock');os.chmod('/socket/proxy.sock',0o666);s.listen();c,_=s.accept();c.sendall(b'pong');c.close()"
        server=d.create(image,['python3','-c',program],run,['--network','bridge','--mount',f'type=volume,src={volume},dst=/socket'])
        d.start(server)
        client="import socket,time; s=socket.socket(socket.AF_UNIX);\nfor i in range(100):\n try:s.connect('/socket/proxy.sock');break\n except OSError:time.sleep(.02)\nprint(s.recv(4).decode())"
        out=d.run(['run','--rm','--label',f'abench.run={run}','--network','none','--user','1000:1000','--cap-drop','ALL','--security-opt','no-new-privileges','--mount',f'type=volume,src={volume},dst=/socket,readonly',image,'python3','-c',client])
        result['readonly_unix_socket']=out.stdout.strip()=='pong'
        out=d.run(['run','--rm','--label',f'abench.run={run}','--network','none','--user','1000:1000',image,'python3','-c',"from pathlib import Path;import json;print(json.dumps({x:Path('/sys/fs/cgroup/'+x).read_text() for x in ['cpu.stat','memory.peak']}))"])
        result['cgroup_v2']=json.loads(out.stdout)
        with tempfile.TemporaryDirectory() as source,tempfile.TemporaryDirectory() as dest:
            (Path(source)/'probe.txt').write_text('tar-roundtrip')
            cid=d.create(image,['python3','-c',"import pathlib,os,json; p=pathlib.Path('/tmp/probe.txt'); print(json.dumps({'uid':p.stat().st_uid,'mode':oct(p.stat().st_mode&0o777),'text':p.read_text()}))"],run,['--network','none','--user','1000:1000'])
            try:
                d.upload(cid,source,'/tmp')
                result['uploaded_owner']=json.loads(d.run(['start','-a',cid]).stdout)
                d.download(cid,'/tmp',dest)
                result['stopped_download']=(Path(dest)/'probe.txt').read_text()=='tar-roundtrip'
            finally:d.remove(cid)
        code="import time; a=time.monotonic();time.sleep(.01);print(time.monotonic()-a)"
        host=[float(subprocess.check_output(['python3','-c',code])) for _ in range(5)]
        child=[float(d.run(['run','--rm','--network','none','--label',f'abench.run={run}',image,'python3','-c',code]).stdout) for _ in range(5)]
        delta=abs(statistics.median(host)-statistics.median(child))
        result['timing']={'host_seconds':host,'container_seconds':child,'median_delta_seconds':delta,'within_10ms':delta<.01,'note':'child measured interval; parent R/T interval is separately retained by stage protocol'}
        result['gpu']='not exercised: only use cards with no active compute users'
        result['claude']='requires locked agent image and independently supplied provider credentials'
        return result
    finally:
        d.gc(run)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--image',required=True);ap.add_argument('--output');a=ap.parse_args()
    result=probe(a.image);text=json.dumps(result,indent=2)+'\n'
    if a.output:Path(a.output).write_text(text)
    print(text)
if __name__=='__main__':main()
