"""Image packaging identity and concurrent local builds, without a Docker daemon or network."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import images
import publish_images


SNAPSHOT = {'schema': 'abench.environment/v1', 'profile': 'cpu-sci',
            'software': {'python': '3.11.13', 'implementation': 'CPython',
                         'distributions': {'numpy': '2.2.6'}, 'torch': None},
            'hardware': {'gpus': []}}


class FakeDocker:
    executable = 'fake-docker'

    def __init__(self, daemon_id='test-daemon'):
        self.daemon_id = daemon_id
        self.images = {'python:3.11-test': {'Id': 'sha256:base'}}
        self.mutex = threading.Lock()
        self.builds = []
        self.parent_tags = []
        self.parent_tag_removals = []
        self.build_parent_ids = []
        self.client_proxy_environments = []
        self.build_failure = self.remove_failure = None
        self.environment_reads = 0
        self.snapshot = deepcopy(SNAPSHOT)

    def info(self):
        return {'platform': 'linux/amd64', 'raw': {'ID': self.daemon_id}}

    def inspect(self, reference):
        with self.mutex:
            if reference not in self.images:
                raise images.DockerError('image not found')
            return deepcopy(self.images[reference])

    def run(self, args, **kwargs):
        if args[:2] == ['context', 'show']:
            return SimpleNamespace(returncode=0, stdout='test-context\n')
        if args[0] == 'tag':
            with self.mutex:
                source = next(item for item in self.images.values() if item['Id'] == args[1])
                self.images[args[2]] = deepcopy(source)
                self.parent_tags.append((args[1], args[2]))
            return SimpleNamespace(returncode=0, stdout='')
        if args[:2] == ['image', 'rm']:
            with self.mutex:
                self.parent_tag_removals.append(args[2])
                if self.remove_failure:
                    return SimpleNamespace(returncode=1, stdout='', stderr=self.remove_failure)
                self.images.pop(args[2])
            return SimpleNamespace(returncode=0, stdout='', stderr='')
        if args[0] == 'build':
            with self.mutex:
                self.builds.append(list(args))
                self.client_proxy_environments.append({name: os.environ.get(name)
                                                       for name in ('HTTP_PROXY', 'HTTPS_PROXY', 'NO_PROXY')})
                index = len(self.builds)
                parent = next((item.split('=', 1)[1] for item in args if item.startswith('PROFILE_IMAGE=')), None)
                if parent:
                    if parent.startswith('sha256:'):
                        raise images.DockerError('BuildKit does not accept a bare image ID in FROM')
                    self.build_parent_ids.append(self.images[parent]['Id'])
            if self.build_failure:
                raise self.build_failure
            time.sleep(.05)  # Give simultaneous requests time to reach the interprocess file lock.
            tag = args[args.index('-t') + 1]
            key, value = args[args.index('--label') + 1].split('=', 1)
            with self.mutex:
                self.images[tag] = {'Id': f'sha256:build{index}', 'Config': {'Labels': {key: value}}}
            return SimpleNamespace(returncode=0, stdout='')
        if args[0] == 'run':
            with self.mutex:
                self.environment_reads += 1
            return SimpleNamespace(returncode=0, stdout=json.dumps(self.snapshot))
        raise AssertionError(args)


class ImageBuildTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / 'repo'
        (self.root / 'spec/env').mkdir(parents=True)
        (self.root / 'tools/runtime').mkdir(parents=True)
        (self.root / 'tools/record_environment.py').write_text('# environment recorder\n')
        (self.root / 'tools/runtime/forwarder.py').write_text('# forwarder\n')
        self.config = {'namespace': 'registry.example/abench', 'node': '24.14.0',
                       'harnesses': {'claude': {'version': '2.1.312', 'npm': '@anthropic-ai/claude-code'}},
                       'images': {'cpu-sci': {'platforms': ['linux/amd64']},
                                  'egress': {'version': '2026.10', 'source': 'python:3.11-test',
                                             'platforms': ['linux/amd64']}}}
        (self.root / 'spec/env/images.yaml').write_text(yaml.safe_dump(self.config))
        (self.root / 'spec/env/profiles.yaml').write_text(yaml.safe_dump({'profiles': {
            'cpu-sci': {'version': '2026.10', 'base': 'python:3.11-test', 'packages': {'numpy': '2.2.6'}}}}))
        (self.root / 'spec/env/images.lock.yaml').write_text('images: {}\n')
        self.root_patch = patch.object(images, 'ROOT', self.root)
        self.root_patch.start()
        self.addCleanup(self.root_patch.stop)
        self.env_patch = patch.dict(os.environ, {'ABENCH_IMAGE_BUILD_CACHE': str(Path(self.tmp.name) / 'cache')})
        self.env_patch.start()
        self.addCleanup(self.env_patch.stop)
        images.generate()

    def write_manifest(self, config):
        (self.root / 'spec/env/images.yaml').write_text(yaml.safe_dump(config))

    def prepare_agent(self, docker):
        path = images.env_lock_path('cpu-sci', '2026.10')
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(SNAPSHOT))
        docker.images['registry.example/abench/cpu-sci:2026.10'] = {'Id': 'sha256:profile-one'}

    def test_agent_versions_include_packaging_and_remain_stable_when_parent_is_published(self):
        original = images.image_version('cpu-sci', 'claude')
        self.assertEqual(images.image_version('cpu-sci'), '2026.10')
        for field, value in [('node', '24.15.0'), ('agent_packaging_revision', 2)]:
            changed = deepcopy(self.config)
            changed[field] = value
            self.write_manifest(changed)
            self.assertNotEqual(images.image_version('cpu-sci', 'claude'), original, field)
        changed = deepcopy(self.config)
        changed['harnesses']['claude']['version'] = '2.1.313'
        self.write_manifest(changed)
        self.assertNotEqual(images.image_version('cpu-sci', 'claude'), original)
        self.write_manifest(self.config)
        dockerfile = images.agent_dockerfile
        with patch.object(images, 'agent_dockerfile', side_effect=lambda *args: dockerfile(*args) + '# recipe change\n'):
            self.assertNotEqual(images.image_version('cpu-sci', 'claude'), original)
        digest = 'sha256:' + 'a' * 64
        with patch.object(images, 'lock_entries', return_value={'cpu-sci@2026.10': {
                'linux/amd64': {'digest': digest, 'reference': 'registry.example/abench/cpu-sci@' + digest}}}):
            self.assertEqual(images.image_version('cpu-sci', 'claude'), original)
        self.assertNotEqual(images.image_version('cpu-sci', 'claude', '2026.09'), original)
        changed = deepcopy(self.config)
        changed['images']['cpu-sci']['platforms'].append('linux/arm64')
        self.write_manifest(changed)
        self.assertNotEqual(images.image_version('cpu-sci', 'claude'), original)
        changed['harnesses']['claude']['source'] = {'pnpm': '10.33.2', 'tag': 'v1.0',
                                                   'repository': 'https://example.invalid/cli', 'commit': 'a' * 40}
        self.write_manifest(changed)
        source_version = images.image_version('cpu-sci', 'claude')
        changed['harnesses']['claude']['source']['commit'] = 'b' * 40
        self.write_manifest(changed)
        self.assertNotEqual(images.image_version('cpu-sci', 'claude'), source_version)

    def test_explicit_historical_agent_version_keeps_its_published_digest(self):
        version = '2026.09-claude2.1.38'
        digest = 'sha256:' + 'b' * 64
        reference = 'registry.example/abench/agent-claude-cpu-sci@' + digest
        with patch.object(images, 'lock_entries', return_value={f'agent-claude-cpu-sci@{version}': {
                'linux/amd64': {'digest': digest, 'reference': reference}}}):
            self.assertEqual(images.resolve_image('cpu-sci', 'claude', version=version,
                                                  platform='linux/amd64'), reference)

    def test_same_concurrent_requests_build_and_verify_only_once(self):
        docker = FakeDocker()
        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(lambda _: images.manage('build', docker=docker), range(4)))
        self.assertEqual(len(docker.builds), 1)
        self.assertEqual(docker.environment_reads, 1)
        self.assertEqual(sum(not result['reused'] for result in results), 1)
        self.assertEqual(len({result['build_inputs_sha256'] for result in results}), 1)
        (self.root / 'case').mkdir()
        (self.root / 'case/task.md').write_text('Changed runtime prompt')
        self.assertTrue(images.manage('build', docker=docker)['reused'])
        self.assertEqual(len(docker.builds), 1)
        (self.root / 'tools/runtime/forwarder.py').write_text('# changed image content\n')
        self.assertFalse(images.manage('build', docker=docker)['reused'])
        self.assertEqual(len(docker.builds), 2)

    def test_unvalidated_or_retagged_images_never_hit_success_cache(self):
        docker = FakeDocker()
        tag = 'registry.example/abench/cpu-sci:2026.10'
        docker.images[tag] = {'Id': 'sha256:old', 'Config': {'Labels': {}}}
        self.assertFalse(images.manage('build', docker=docker)['reused'])
        docker.images[tag]['Id'] = 'sha256:replaced'
        self.assertFalse(images.manage('build', docker=docker)['reused'])
        self.assertEqual(len(docker.builds), 2)

    def test_failed_environment_check_has_no_success_marker(self):
        docker = FakeDocker()
        self.prepare_agent(docker)
        docker.snapshot['software']['distributions']['numpy'] = 'wrong'
        with self.assertRaisesRegex(ValueError, 'modified Python environment'):
            images.manage('build', agent='claude', docker=docker)
        self.assertEqual(list((Path(self.tmp.name) / 'cache').rglob('*.json')), [])
        docker.snapshot = deepcopy(SNAPSHOT)
        self.assertFalse(images.manage('build', agent='claude', docker=docker)['reused'])
        self.assertTrue(images.manage('build', agent='claude', docker=docker)['reused'])
        self.assertEqual(len(docker.builds), 2)

    def test_actual_parent_identity_and_daemon_separate_build_reuse(self):
        docker = FakeDocker()
        self.prepare_agent(docker)
        first = images.manage('build', agent='claude', docker=docker)
        docker.images['registry.example/abench/cpu-sci:2026.10']['Id'] = 'sha256:profile-two'
        second = images.manage('build', agent='claude', docker=docker)
        self.assertEqual(first['image'], second['image'])
        self.assertNotEqual(first['build_inputs_sha256'], second['build_inputs_sha256'])
        self.assertEqual(docker.build_parent_ids, ['sha256:profile-one', 'sha256:profile-two'])
        self.assertTrue(all(tag.startswith('abench-build-parent:') for _, tag in docker.parent_tags))
        self.assertEqual(len({tag for _, tag in docker.parent_tags}), 2)
        self.assertEqual(docker.parent_tag_removals, [tag for _, tag in docker.parent_tags])
        self.assertFalse(any(tag in docker.images for _, tag in docker.parent_tags))
        self.assertEqual(docker.images['registry.example/abench/cpu-sci:2026.10']['Id'], 'sha256:profile-two')
        another = FakeDocker('another-daemon')
        self.prepare_agent(another)
        self.assertFalse(images.manage('build', agent='claude', docker=another)['reused'])
        self.assertEqual(len(another.builds), 1)

    def test_concurrent_agent_builds_share_work_without_sharing_a_temporary_parent_tag(self):
        docker = FakeDocker()
        self.prepare_agent(docker)
        with ThreadPoolExecutor(max_workers=3) as pool:
            results = list(pool.map(lambda _: images.manage('build', agent='claude', docker=docker), range(3)))
        self.assertEqual(len(docker.builds), 1)
        self.assertEqual(len(docker.parent_tags), 1)
        self.assertEqual(docker.build_parent_ids, ['sha256:profile-one'])
        self.assertEqual(docker.parent_tag_removals, [docker.parent_tags[0][1]])
        self.assertEqual(sum(not result['reused'] for result in results), 1)

    def test_failed_build_removes_only_its_parent_tag_and_preserves_original_failure(self):
        for cleanup_failure in (None, 'daemon unavailable'):
            with self.subTest(cleanup_failure=cleanup_failure):
                docker = FakeDocker()
                self.prepare_agent(docker)
                docker.build_failure = images.DockerError('original build failure')
                docker.remove_failure = cleanup_failure
                with self.assertRaisesRegex(images.DockerError, 'original build failure') as caught:
                    images.manage('build', agent='claude', docker=docker)
                self.assertEqual(docker.parent_tag_removals, [docker.parent_tags[0][1]])
                self.assertIn('registry.example/abench/cpu-sci:2026.10', docker.images)
                self.assertEqual(list((Path(self.tmp.name) / 'cache').rglob('*.json')), [])
                if cleanup_failure:
                    self.assertTrue(any('daemon unavailable' in note for note in getattr(caught.exception, '__notes__', [])))
                else:
                    self.assertNotIn(docker.parent_tags[0][1], docker.images)

    def test_parent_tag_cleanup_failure_after_build_does_not_write_success_marker(self):
        docker = FakeDocker()
        self.prepare_agent(docker)
        docker.remove_failure = 'temporary tag busy'
        with self.assertRaisesRegex(images.DockerError, 'failed to remove temporary parent tag'):
            images.manage('build', agent='claude', docker=docker)
        self.assertEqual(list((Path(self.tmp.name) / 'cache').rglob('*.json')), [])
        self.assertEqual(docker.environment_reads, 0)

    def test_build_proxy_override_preserves_host_proxy_and_does_not_change_content_identity(self):
        docker = FakeDocker()
        host = {'HTTP_PROXY': 'http://127.0.0.1:8080', 'HTTPS_PROXY': 'http://127.0.0.1:8080',
                'NO_PROXY': 'localhost,127.0.0.1'}
        with patch.dict(os.environ, {**host, 'ABENCH_BUILD_PROXY': 'http://10.0.2.2:8080',
                                    'ABENCH_BUILD_NETWORK': 'host'}):
            version = images.image_version('cpu-sci', 'claude')
            first = images.manage('build', docker=docker)
            args = docker.builds[0]
            self.assertIn('HTTP_PROXY=http://10.0.2.2:8080', args)
            self.assertIn('HTTPS_PROXY=http://10.0.2.2:8080', args)
            self.assertIn('NO_PROXY', args)
            self.assertEqual(docker.client_proxy_environments, [host])
            self.assertEqual({name: os.environ[name] for name in host}, host)
            with patch.dict(os.environ, {'ABENCH_BUILD_PROXY': 'http://another-build-proxy:8080',
                                        'HTTPS_PROXY': 'http://another-host-proxy:8080'}):
                (self.root / 'task.md').write_text('An unrelated runtime prompt')
                second = images.manage('build', docker=docker)
                self.assertTrue(second['reused'])
                self.assertEqual(second['build_inputs_sha256'], first['build_inputs_sha256'])
                self.assertEqual(images.image_version('cpu-sci', 'claude'), version)
        self.assertEqual(len(docker.builds), 1)

    def test_default_proxy_build_arguments_keep_existing_environment_passthrough(self):
        docker = FakeDocker()
        host = {'HTTP_PROXY': 'http://host-proxy:8080', 'HTTPS_PROXY': 'http://host-proxy:8080',
                'NO_PROXY': 'localhost,127.0.0.1'}
        with patch.dict(os.environ, {**host, 'ABENCH_BUILD_PROXY': ''}):
            images.manage('build', docker=docker)
        args = docker.builds[0]
        self.assertEqual(docker.client_proxy_environments, [host])
        for name in host:
            self.assertIn(name, args)
            self.assertFalse(any(item.startswith(name + '=') for item in args))

    def publication_lock(self, recipe=None):
        digest = 'sha256:' + 'c' * 64
        parent = {'digest': digest, 'reference': 'registry.example/abench/cpu-sci@' + digest}
        lock = {'images': {'cpu-sci@2026.10': {'linux/amd64': parent},
                           'egress@2026.10': {'linux/amd64': parent}}}
        if recipe is not None:
            version = images.image_version('cpu-sci', 'claude')
            lock['images'][f'agent-claude-cpu-sci@{version}'] = {
                'linux/amd64': {**parent, 'recipe_sha256': recipe}}
        return lock

    def test_publication_does_not_skip_a_mismatched_recipe_lock(self):
        path = self.root / 'spec/env/images.lock.yaml'
        path.write_text(yaml.safe_dump(self.publication_lock('wrong')))
        with patch.object(publish_images, 'LOCK', path), patch.object(publish_images, 'published', return_value=True), \
                patch.object(publish_images, 'command') as command:
            with self.assertRaisesRegex(RuntimeError, 'packaging identity'):
                publish_images.main()
        command.assert_not_called()

    def test_node_upgrade_releases_a_new_agent_tag_and_preserves_old_lock(self):
        old_version = images.image_version('cpu-sci', 'claude')
        old_recipe = images.agent_recipe_hash('cpu-sci', 'claude')
        lock = self.publication_lock(old_recipe)
        path = self.root / 'spec/env/images.lock.yaml'
        path.write_text(yaml.safe_dump(lock))
        changed = deepcopy(self.config)
        changed['node'] = '24.15.0'
        self.write_manifest(changed)
        version = images.image_version('cpu-sci', 'claude')
        images.env_lock_path('cpu-sci', '2026.10').parent.mkdir(parents=True, exist_ok=True)
        images.env_lock_path('cpu-sci', '2026.10').write_text(json.dumps(SNAPSHOT))
        agent_tag = f'registry.example/abench/agent-claude-cpu-sci:{version}'
        def is_published(tag):
            return tag != agent_tag
        def command(args, capture=False):
            return json.dumps(SNAPSHOT) if args[:2] == ['docker', 'run'] else ''
        new_entry = {'linux/amd64': {'digest': 'sha256:' + 'd' * 64,
                                    'reference': 'registry.example/abench/agent-claude-cpu-sci@sha256:' + 'd' * 64}}
        with patch.object(publish_images, 'LOCK', path), patch.object(publish_images, 'published', side_effect=is_published), \
                patch.object(publish_images, 'command', side_effect=command) as run, \
                patch.object(publish_images, 'platform_locks', return_value=new_entry), \
                patch.object(publish_images.subprocess, 'run'):
            publish_images.main()
        pushes = [call.args[0] for call in run.call_args_list if '--push' in call.args[0]]
        self.assertEqual(len(pushes), 1)
        self.assertIn(agent_tag, pushes[0])
        saved = yaml.safe_load(path.read_text())['images']
        self.assertIn(f'agent-claude-cpu-sci@{old_version}', saved)
        self.assertEqual(saved[f'agent-claude-cpu-sci@{version}']['linux/amd64']['recipe_sha256'],
                         images.agent_recipe_hash('cpu-sci', 'claude'))


if __name__ == '__main__':
    unittest.main()
