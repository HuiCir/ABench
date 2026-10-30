"""Opt-in agent isolation/lifecycle tests using only a local CPU image and no API key."""
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace
import unittest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
from agents.base import Launch
from runtime.agent import DockerAgentEnvironment
from egress_proxy import policy_for_case, summarize_log


@unittest.skipUnless(os.environ.get('ABENCH_TEST_IMAGE') and shutil.which('docker'),
                     'set ABENCH_TEST_IMAGE to a locally available CPU profile image')
class DockerAgentTest(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='abench-test-agent-')
        self.root=Path(self.tmp.name)
        self.paths={'root':self.root,'view':self.root/'view','workspace':self.root/'workspace','control':self.root/'control'}
        for name in ('view','workspace','control'):
            self.paths[name].mkdir()
        (self.paths['view']/'task.md').write_text('test task\n')
        self.args=SimpleNamespace(case='TC-FT-001',run_id='integration-test',gpus=None,network='none')
        self.oj=yaml.safe_load((ROOT/'cases/TC-FT-001/case/oj.yaml').read_text())
        self.environment=None

    def tearDown(self):
        if self.environment:
            self.environment.close()
        self.tmp.cleanup()

    def start(self,proxy=False):
        pol=None
        if proxy:
            self.args.network='egress'
            pol=policy_for_case({},'mock')
            for src,name in ((ROOT/'tools/egress_proxy.py','egress_proxy.py'),
                             (ROOT/'tools/runtime/forwarder.py','forwarder.py')):
                shutil.copyfile(src,self.paths['control']/name)
            (self.paths['control']/'policy.json').write_text(json.dumps(pol))
        image=os.environ['ABENCH_TEST_IMAGE']
        self.environment=DockerAgentEnvironment(self.args,self.paths,self.oj,image,image,pol)
        return self.environment.start()

    def execute(self,code,timeout=10,env=None):
        return self.environment.execute(Launch(['python3','-c',code]),{'HOME':'/abench/workspace/.home',**(env or {})},timeout,
                                        self.root/'out.log',self.root/'err.log')

    def test_network_dns_mounts_and_denied_hub_proxy(self):
        rec=self.start(proxy=True)
        self.assertTrue(rec['verified'],rec)
        self.assertTrue(rec['probes']['direct_blocked'])
        self.assertTrue(rec['probes']['dns_blocked'])
        self.assertTrue(rec['probes']['view_readonly'])
        code="""import urllib.request
try:
 urllib.request.urlopen('https://huggingface.co',timeout=5)
 raise RuntimeError('hub unexpectedly reachable')
except urllib.error.URLError as e:
 assert '403' in str(e), str(e)
open('/abench/workspace/submission/result.txt','w').write('blocked')
"""
        status,code=self.execute(code,env={'HTTPS_PROXY':'http://127.0.0.1:3128','NO_PROXY':'localhost,127.0.0.1'})
        self.assertEqual(status,'succeeded',(status,code,(self.root/'err.log').read_text()))
        result=self.environment.collect()
        self.assertEqual((result['submission']/'result.txt').read_text(),'blocked')
        summary=summarize_log(result['egress_log'])
        self.assertTrue(summary['chain_ok'])
        self.assertGreaterEqual(summary['denied'],1)

    def test_timeout_is_accounted_then_killed(self):
        self.start()
        status,_=self.execute('import time; time.sleep(30)',timeout=1)
        self.assertEqual(status,'timeout')
        # collect() reads the container cgroup first, so a timed-out run still has CPU accounting, then kills it
        self.environment.collect()
        self.assertFalse(self.environment.docker.inspect(self.environment.cid)['State']['Running'])
        self.assertIsNotNone(self.environment.record['cpu_usage_usec'])

    def test_collection_rejects_symlink(self):
        self.start()
        self.assertEqual(self.execute("import os; os.symlink('/etc/passwd','/abench/workspace/submission/stolen')")[0],'succeeded')
        with self.assertRaises(ValueError):
            self.environment.collect()
