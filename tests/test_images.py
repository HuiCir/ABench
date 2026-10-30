import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import images
class ImageTests(unittest.TestCase):
    def test_unknown_version_rejected(self):
        with self.assertRaises(ValueError):images.resolve_image('cpu-sci',version='never',platform='linux/amd64')
    def test_unpublished_image_never_scores(self):
        config={'namespace':'registry.example/abench','images':{}}
        with patch.object(images,'manifest',return_value=config), patch.object(images,'lock_entries',return_value={}):
            with self.assertRaises(ValueError):images.resolve_image('cpu-sci',platform='linux/amd64')
            with patch.dict(os.environ,{'ABENCH_IMAGE_CPU_SCI':'local:test'}):
                with self.assertRaises(ValueError):images.resolve_image('cpu-sci',platform='linux/amd64')
                self.assertEqual(images.resolve_image('cpu-sci',platform='linux/amd64',allow_unlocked=True),'local:test')
    def test_profile_parent_preserves_python(self):
        p=images.profiles()
        self.assertEqual(len(p),7)
        for name,item in p.items():
            self.assertEqual(item['version'],'2026.10')
            for version in item['packages'].values():self.assertNotIn('>',version)
        text=(images.ROOT/'docker/profiles/gpu-torch-hf/Dockerfile').read_text()
        self.assertIn('FROM ghcr.io/huicir/abench/gpu-torch:2026.10',text)
        self.assertIn('COPY --from=cuda /usr/local/cuda',(images.ROOT/'docker/profiles/gpu-torch-cuda-dev/Dockerfile').read_text())
    def test_versions_are_locked_side_by_side(self):
        old='sha256:'+'1'*64;new='sha256:'+'2'*64;agent='sha256:'+'3'*64
        lock={'cpu-sci@2026.09':{'linux/amd64':{'digest':old,'reference':'ghcr.io/huicir/abench/cpu-sci@'+old}},
              'cpu-sci@2026.10':{'linux/amd64':{'digest':new,'reference':'ghcr.io/huicir/abench/cpu-sci@'+new}},
              'agent-claude-cpu-sci@'+images.image_version('cpu-sci','claude'):{'linux/amd64':{'digest':agent,'reference':'ghcr.io/huicir/abench/agent-claude-cpu-sci@'+agent}}}
        with patch.object(images,'lock_entries',return_value=lock):
            self.assertTrue(images.resolve_image({'profile':'cpu-sci','version':'2026.09'},platform='linux/amd64').endswith(old))
            self.assertTrue(images.resolve_image('cpu-sci',platform='linux/amd64').endswith(new))
            self.assertTrue(images.resolve_image('cpu-sci@2026.10','claude',platform='linux/amd64').endswith(agent))
            self.assertEqual(images.known_versions('cpu-sci'),{'2026.09','2026.10'})
            with self.assertRaisesRegex(ValueError,'unknown profile version'):
                images.resolve_image({'profile':'cpu-sci','version':'2026.09'},'claude',platform='linux/amd64')
        with patch.object(images,'lock_entries',return_value={}):
            self.assertEqual(images.resolve_image('cpu-sci','claude',platform='linux/amd64',allow_unlocked=True),
                             'ghcr.io/huicir/abench/agent-claude-cpu-sci:'+images.image_version('cpu-sci','claude'))
            with self.assertRaisesRegex(ValueError,'unknown profile version'):
                images.resolve_image({'profile':'cpu-sci','version':'2026.09'},platform='linux/amd64',allow_unlocked=True)
        self.assertEqual(images.env_lock_path('cpu-sci','2026.10','linux/arm64').name,'cpu-sci@2026.10-linux-arm64.json')
        for name in images.profiles():
            if name in ('cpu-sci','gpu-torch'):
                self.assertTrue(images.env_lock_path(name,images.image_version(name)).is_file(),name)
    def test_lock_is_immutable_per_version_and_new_versions_add_entries(self):
        import shutil,tempfile
        digest=lambda c:'sha256:'+c*64
        class FakeDocker:
            def __init__(self,ref):self.ref=ref
            def info(self):return {'platform':'linux/amd64'}
            def inspect(self,tag):return {'RepoDigests':[self.ref]}
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'spec/env').mkdir(parents=True)
            shutil.copyfile(images.ROOT/'spec/env/images.yaml',root/'spec/env/images.yaml')
            shutil.copyfile(images.ROOT/'spec/env/profiles.yaml',root/'spec/env/profiles.yaml')
            (root/'spec/env/images.lock.yaml').write_text('schema: abench.images_lock/v1\nimages: {}\n')
            with patch.object(images,'ROOT',root):
                ref='ghcr.io/huicir/abench/cpu-sci@'+digest('a')
                images.manage('lock','cpu-sci',docker=FakeDocker(ref))
                with self.assertRaisesRegex(ValueError,'immutable'):
                    images.manage('lock','cpu-sci',docker=FakeDocker('ghcr.io/huicir/abench/cpu-sci@'+digest('b')))
                text=(root/'spec/env/profiles.yaml').read_text().replace("version: '2026.10'","version: '2026.11'",1)
                (root/'spec/env/profiles.yaml').write_text(text)
                images.manage('lock','cpu-sci',docker=FakeDocker('ghcr.io/huicir/abench/cpu-sci@'+digest('b')))
                self.assertEqual(set(images.lock_entries()),{'cpu-sci@2026.10','cpu-sci@2026.11'})
                self.assertEqual(images.known_versions('cpu-sci'),{'2026.10','2026.11'})
    def test_every_harness_has_one_pinned_agent_dockerfile(self):
        from agents import HARNESSES,get_agent
        for name,pin in images.manifest()['harnesses'].items():
            self.assertIn(name,HARNESSES)
            self.assertEqual(get_agent(name).version,str(pin['version']))
            text=(images.ROOT/f'docker/agents/{name}/Dockerfile').read_text()
            self.assertIn('FROM ${PROFILE_IMAGE}',text)
            self.assertIn(f"{pin['npm']}@{pin['version']}" if 'npm' in pin else pin['source']['commit'],text)
            self.assertTrue(images.image_version('gpu-torch',name).startswith(f"2026.10-{name}{pin['version']}-b"))
if __name__=='__main__':unittest.main()
