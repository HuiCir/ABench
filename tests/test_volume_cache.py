"""Cache integrity, cold recovery and source reuse without requiring a Docker daemon."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
import uuid
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from runtime import cache
from runtime.transport import Docker, DockerError
from agents import workspace


class FakeDocker(Docker):
    """Volume operations use real temporary files; the verifier runs its real streaming hash code."""
    def __init__(self, root, backend=None, daemon='engine-one', context='default'):
        super().__init__()
        self.root = Path(root)
        self.backend = backend if backend is not None else {'volumes': {}, 'helpers': {}, 'uploads': 0, 'verifies': 0}
        self.daemon, self.context = daemon, context

    def info(self):
        return {'raw': {'ID': self.daemon, 'DockerRootDir': '/var/lib/docker'}}

    def run(self, args, check=True, **kwargs):
        if args[:2] == ['context', 'show']:
            return SimpleNamespace(returncode=0, stdout=self.context, stderr='')
        if args[:2] == ['context', 'inspect']:
            return SimpleNamespace(returncode=0, stdout=json.dumps([{
                'Endpoints': {'docker': {'Host': 'unix:///run/docker.sock'}}}]), stderr='')
        if args[:2] == ['volume', 'inspect']:
            if self.backend.get('inspect_failure'):
                return SimpleNamespace(returncode=1, stdout='', stderr='Cannot connect to Docker daemon')
            entry = self.backend['volumes'].get((self.daemon, args[-1]))
            return SimpleNamespace(returncode=0 if entry else 1,
                                   stdout=json.dumps([entry['meta']]) if entry else '',
                                   stderr='' if entry else 'Error response from daemon: no such volume')
        raise AssertionError(args)

    def volume(self, name, run_id, labels=None):
        key = (self.daemon, name)
        if key not in self.backend['volumes']:
            directory = self.root / self.daemon / name
            directory.mkdir(parents=True)
            self.backend['volumes'][key] = {'path': directory, 'meta': {
                'Name': name, 'Driver': 'local', 'CreatedAt': '2026-10-29T00:00:00Z',
                'Labels': {'abench.run': run_id, **(labels or {})}, 'Mountpoint': str(directory)}}
        return name

    def create(self, image, command, run_id, options=None, labels=None):
        name = next(x.split('src=', 1)[1].split(',', 1)[0] for x in options if 'src=' in x)
        cid = 'helper-' + str(len(self.backend['helpers']))
        self.backend['helpers'][cid] = {'volume': name, 'options': options}
        return cid

    def start(self, cid):
        self.assert_helper(cid)

    def assert_helper(self, cid):
        if cid not in self.backend['helpers']:
            raise AssertionError(cid)

    def upload(self, cid, source, destination):
        self.backend['uploads'] += 1
        if self.backend.get('upload_failure'):
            raise DockerError('injected upload failure')
        target = self.backend['volumes'][(self.daemon, self.backend['helpers'][cid]['volume'])]['path']
        source = Path(source)
        if source.is_dir():
            shutil.copytree(source, target, dirs_exist_ok=True)
        else:
            shutil.copy2(source, target / source.name)
        time.sleep(self.backend.get('upload_delay', 0))
        if self.backend.get('after_upload'):
            self.backend['after_upload'](source)

    def exec(self, cid, command, **kwargs):
        helper = self.backend['helpers'][cid]
        if not any('readonly' in value for value in helper['options']):
            raise AssertionError('cold verification must use a read-only mount')
        root = self.backend['volumes'][(self.daemon, helper['volume'])]['path']
        self.backend['verifies'] += 1
        script = command[-1].replace("pathlib.Path('/cache')", f'pathlib.Path({str(root)!r})')
        result = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True, check=True)
        return result

    def remove(self, cid):
        if self.backend.get('remove_failure'):
            return SimpleNamespace(returncode=1, stdout='', stderr='injected rm failure')
        self.backend['helpers'].pop(cid, None)
        return SimpleNamespace(returncode=0, stdout='', stderr='')

    def remove_volume(self, name):
        entry = self.backend['volumes'].pop((self.daemon, name), None)
        if entry:
            shutil.rmtree(entry['path'])
        return SimpleNamespace(returncode=0, stdout='', stderr='')


class VolumeCacheTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.source = self.root / 'source'
        self.source.mkdir()
        (self.source / 'data.npz').write_bytes(b'first data')
        self.patch = patch.dict(os.environ, {'ABENCH_CACHE_DIR': str(self.root / 'persistent')})
        self.patch.start()
        self.docker = FakeDocker(self.root / 'docker')

    def tearDown(self):
        self.patch.stop()
        self.temp.cleanup()

    def ready(self):
        return list((self.root / 'persistent').rglob('*.ready.json'))

    def test_same_source_hashes_once_per_run_and_normal_changes_invalidate_memo(self):
        with patch('runtime.cache._file_hash', wraps=cache._file_hash) as hashing:
            first = self.docker.cached_volume(self.source, 'cpu')
            self.assertEqual(self.docker.cached_volume(self.source, 'gpu'), first)
            self.assertEqual(hashing.call_count, 1)
            self.assertEqual(self.docker.backend['uploads'], 1)
            self.assertTrue(self.docker.cache_records[-1]['source_memo_hit'])
            file = self.source / 'data.npz'
            stamp = file.stat()
            file.write_bytes(b'other data')                  # same length and restored mtime still changes ctime
            os.utime(file, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
            second = self.docker.cached_volume(self.source, 'cpu')
            self.assertNotEqual(second, first)
            self.assertEqual(hashing.call_count, 2)
            self.assertEqual(self.docker.backend['uploads'], 2)

    def test_lost_ready_record_verifies_existing_content_and_reuses_without_upload(self):
        name = self.docker.cached_volume(self.source, 'cpu')
        self.ready()[0].unlink()
        other = FakeDocker(self.root / 'docker', self.docker.backend)
        self.assertEqual(other.cached_volume(self.source, 'cpu'), name)
        self.assertEqual(other.backend['uploads'], 1)
        self.assertEqual(other.backend['verifies'], 1)
        self.assertTrue(other.cache_records[0]['cold_verified'])
        self.assertTrue(other.cache_records[0]['hit'])

    def test_executable_permission_change_gets_a_different_cache(self):
        source = self.source / 'data.npz'
        first = self.docker.cached_volume(self.source, 'cpu')
        source.chmod(source.stat().st_mode | 0o100)
        self.assertNotEqual(self.docker.cached_volume(self.source, 'cpu'), first)

    def test_changed_volume_identity_forces_verification_and_tampering_is_not_deleted(self):
        name = self.docker.cached_volume(self.source, 'cpu')
        entry = self.docker.backend['volumes'][(self.docker.daemon, name)]
        entry['meta']['CreatedAt'] = '2026-10-29T00:00:01Z'
        (entry['path'] / 'data.npz').write_bytes(b'tampered')
        with self.assertRaisesRegex(DockerError, 'content does not match'):
            self.docker.cached_volume(self.source, 'cpu')
        self.assertIn((self.docker.daemon, name), self.docker.backend['volumes'])
        self.assertEqual((entry['path'] / 'data.npz').read_bytes(), b'tampered')
        self.assertEqual(self.docker.backend['uploads'], 1)

    def test_foreign_ownership_is_rejected_without_deleting_or_reading_the_volume(self):
        name = self.docker.cached_volume(self.source, 'cpu')
        entry = self.docker.backend['volumes'][(self.docker.daemon, name)]
        entry['meta']['Labels'] = {'owner': 'someone-else'}
        with self.assertRaisesRegex(DockerError, 'ownership'):
            self.docker.cached_volume(self.source, 'cpu')
        self.assertIn((self.docker.daemon, name), self.docker.backend['volumes'])
        self.assertEqual(self.docker.backend['verifies'], 0)

    def test_same_content_on_two_daemons_has_separate_ready_records(self):
        first = self.docker.cached_volume(self.source, 'cpu')
        other = FakeDocker(self.root / 'docker', self.docker.backend, daemon='engine-two')
        self.assertEqual(other.cached_volume(self.source, 'cpu'), first)
        self.assertEqual(len(self.ready()), 2)
        self.assertEqual(other.backend['uploads'], 2)

    def test_concurrent_requests_upload_the_same_content_only_once(self):
        self.docker.backend['upload_delay'] = .05
        other = FakeDocker(self.root / 'docker', self.docker.backend)
        with ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(self.docker.cached_volume, self.source, 'cpu')
            second = pool.submit(other.cached_volume, self.source, 'gpu')
            self.assertEqual(first.result(), second.result())
        self.assertEqual(self.docker.backend['uploads'], 1)
        self.assertEqual(self.docker.backend['verifies'], 0)

    def test_upload_or_source_change_never_publishes_ready(self):
        for failure in ('upload_failure', 'after_upload'):
            with self.subTest(failure=failure):
                if failure == 'upload_failure':
                    self.docker.backend[failure] = True
                else:
                    self.docker.backend[failure] = lambda source: (source / 'data.npz').write_bytes(b'changed')
                with self.assertRaises(DockerError):
                    self.docker.cached_volume(self.source, 'cpu')
                self.assertEqual(self.ready(), [])
                self.assertEqual(self.docker.backend['volumes'], {})
                self.assertEqual(self.docker.backend['helpers'], {})
                self.docker.backend.pop(failure)

    def test_cold_verifier_cleanup_failure_is_reported_and_does_not_publish_ready(self):
        name = self.docker.cached_volume(self.source, 'cpu')
        self.ready()[0].unlink()
        self.docker.backend['remove_failure'] = True
        with self.assertRaisesRegex(DockerError, 'cleanup failed'):
            self.docker.cached_volume(self.source, 'cpu')
        self.assertEqual(self.ready(), [])
        self.assertIn((self.docker.daemon, name), self.docker.backend['volumes'])

    def test_daemon_error_is_not_treated_as_a_missing_volume(self):
        self.docker.backend['inspect_failure'] = True
        with self.assertRaisesRegex(DockerError, 'cannot inspect'):
            self.docker.cached_volume(self.source, 'cpu')
        self.assertEqual(self.docker.backend['volumes'], {})
        self.assertEqual(self.ready(), [])

    def test_public_files_use_verified_hashes_and_reject_changes_and_unsafe_names(self):
        source = self.source / 'data.npz'
        item = {'name': source.name, 'source': str(source), 'sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                'verified_stat': cache.file_stamp(source)}
        with patch('runtime.cache._file_hash', wraps=cache._file_hash) as hashing:
            name = self.docker.cached_file_volume([item], 'cpu')
            self.assertEqual(hashing.call_count, 0)
        stored = self.docker.backend['volumes'][(self.docker.daemon, name)]['path'] / source.name
        self.assertEqual(stored.read_bytes(), source.read_bytes())
        for invalid in (['../data.npz'], ['data.npz', 'data.npz']):
            with self.subTest(names=invalid), self.assertRaises(ValueError):
                self.docker.cached_file_volume([dict(item, name=value) for value in invalid], 'cpu')
        source.write_bytes(b'changed')
        with self.assertRaisesRegex(DockerError, 'changed after validation'):
            self.docker.cached_file_volume([item], 'cpu')


class AgentPublicViewTests(unittest.TestCase):
    def test_duplicate_or_traversing_public_paths_are_rejected(self):
        original = workspace.ROOT
        for paths in (['public/data.npz', 'public/data.npz'], ['../outside'], ['/absolute']):
            with self.subTest(paths=paths), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                task = root / 'cases/TEST/case/agent/task.md'
                task.parent.mkdir(parents=True)
                task.write_text('Train using the supplied public data.')
                templates = root / 'sandbox/templates'
                templates.mkdir(parents=True)
                for name in ('design_report.html', 'references.yaml'):
                    shutil.copyfile(original / 'sandbox/templates' / name, templates / name)
                data = root / 'public/data.npz'
                data.parent.mkdir()
                data.write_bytes(b'public bytes')
                digest = hashlib.sha256(data.read_bytes()).hexdigest()
                oj = {'data': {'public': [{'path': path, 'sha256': digest} for path in paths]}}
                with patch('agents.workspace.ROOT', root), self.assertRaises(SystemExit):
                    workspace.build_agent_view('TEST', oj, root / 'view', copy_data=False)

    def test_copy_and_mount_views_keep_names_and_frozen_hashes_without_rehashing_data(self):
        original = workspace.ROOT
        for copy_data in (True, False):
            with self.subTest(copy_data=copy_data), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                task = root / 'cases/TEST/case/agent/task.md'
                task.parent.mkdir(parents=True)
                task.write_text('Train using the supplied public data.')
                templates = root / 'sandbox/templates'
                templates.mkdir(parents=True)
                for name in ('design_report.html', 'references.yaml'):
                    shutil.copyfile(original / 'sandbox/templates' / name, templates / name)
                data = root / 'public/data.npz'
                data.parent.mkdir()
                data.write_bytes(b'public bytes')
                digest = hashlib.sha256(data.read_bytes()).hexdigest()
                oj = {'data': {'public': [{'path': 'public/data.npz', 'sha256': digest}]}}
                view = root / 'view'
                with patch('agents.workspace.ROOT', root), \
                        patch('agents.workspace.sha256_file', wraps=workspace.sha256_file) as hashing:
                    manifest = workspace.build_agent_view('TEST', oj, view, copy_data=copy_data)
                self.assertEqual(sum(call.args[0] == data for call in hashing.call_args_list), 1)
                self.assertFalse(any(call.args[0] == view / 'data/data.npz' for call in hashing.call_args_list))
                self.assertEqual((view / 'data/data.npz').read_bytes(), b'public bytes' if copy_data else b'')
                self.assertEqual(manifest['files']['data/data.npz'], digest)
                self.assertEqual(manifest['public_sources'][0]['source'], str(data))
                paths = {key: root / key for key in ('workspace', 'checkpoint', 'submission')}
                self.assertIn('data/data.npz', workspace.agent_prompt({'view': view, **paths}, network='none'))


@unittest.skipUnless(os.environ.get('ABENCH_TEST_IMAGE'), 'set ABENCH_TEST_IMAGE for real Docker cache tests')
class DockerVolumeCacheTests(unittest.TestCase):
    def test_directory_cold_recovery_public_files_and_read_only_mount(self):
        image = os.environ['ABENCH_TEST_IMAGE']
        docker = Docker()
        volumes, helpers = set(), set()
        with tempfile.TemporaryDirectory() as raw, patch.dict(os.environ, {'ABENCH_CACHE_DIR': raw + '/metadata'}):
            root = Path(raw)
            source = root / 'frozen'
            source.mkdir()
            token = uuid.uuid4().hex               # Its content makes these caches unique to this test.
            (source / 'first.txt').write_text(token)
            (source / 'first.txt').chmod(0o755)      # Cold verification also checks preserved executable bits.
            (source / 'nested').mkdir()
            (source / 'nested/second.txt').write_text(token + '-nested')
            public = root / 'allowed.txt'
            public.write_text(token + '-public')
            try:
                name = docker.cached_volume(source, image)
                volumes.add(name)
                self.assertFalse(docker.cache_records[-1]['hit'])
                with patch('runtime.cache._file_hash', side_effect=AssertionError('source was reread')):
                    self.assertEqual(docker.cached_volume(source, image), name)
                self.assertTrue(docker.cache_records[-1]['source_memo_hit'])
                for ready in (root / 'metadata').rglob('*.ready.json'):
                    ready.unlink()
                other = Docker()
                with patch.object(other, 'upload', side_effect=AssertionError('cold recovery reuploaded data')):
                    self.assertEqual(other.cached_volume(source, image), name)
                self.assertTrue(other.cache_records[-1]['cold_verified'])
                self.assertTrue(other.cache_records[-1]['hit'])

                file_volume = docker.cached_file_volume([{'name': public.name, 'source': str(public),
                    'sha256': hashlib.sha256(public.read_bytes()).hexdigest(),
                    'verified_stat': cache.file_stamp(public)}], image)
                volumes.add(file_volume)
                helper = docker.create(image, ['sleep', 'infinity'], 'cache-test-' + token,
                    ['--network', 'none', '--cap-drop', 'ALL', '--security-opt', 'no-new-privileges',
                     '--user', '1000:1000', '--mount', f'type=volume,src={name},dst=/data,readonly',
                     '--mount', f'type=volume,src={file_volume},dst=/public,readonly'])
                helpers.add(helper)
                docker.start(helper)
                script = '''import json,pathlib
out={'first':pathlib.Path('/data/first.txt').read_text(),
     'nested':pathlib.Path('/data/nested/second.txt').read_text(),
     'public':pathlib.Path('/public/allowed.txt').read_text()}
try: pathlib.Path('/public/new').write_text('blocked'); out['readonly']=False
except OSError: out['readonly']=True
print(json.dumps(out))
'''
                result = json.loads(docker.exec(helper, ['python3', '-c', script]).stdout)
                self.assertEqual(result, {'first': token, 'nested': token + '-nested',
                                          'public': token + '-public', 'readonly': True})
            finally:
                # No global gc/prune; the UUID content and returned names identify only this test's objects.
                for helper in helpers:
                    result = docker.remove(helper)
                    self.assertEqual(result.returncode, 0, result.stderr)
                for name in volumes:
                    result = docker.remove_volume(name)
                    self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == '__main__':
    unittest.main()
