"""Content-addressed, read-only data volumes and run-local source hashing.

The persistent ready record binds a verified volume to the daemon and its creation identity. Losing that
record costs one read of the volume, not another upload. These data/model caches include relative paths,
permission bits and file bytes in their identity: changing executable permissions produces a new cache too.
Source memoization lasts only for this Docker object and is invalidated by filesystem metadata changes;
it is not a persistent substitute for content validation.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
import time
import uuid

from runtime.transport import DockerError


def file_stamp(path):
    p = Path(path)
    s = p.lstat()
    if not stat.S_ISREG(s.st_mode):
        raise ValueError(f'cache source must be a regular file: {p}')
    return [s.st_dev, s.st_ino, s.st_mode, s.st_size, s.st_mtime_ns, s.st_ctime_ns]


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True)


def _file_hash(path):
    h = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def _digest(entries):
    return hashlib.sha256(_json(entries).encode()).hexdigest()


# Keep this small self-contained verifier independent of the installed ABench code or agent workspace.
_VERIFY = '''import hashlib,json,os,pathlib,stat
root=pathlib.Path('/cache')
entries=[]
for path in sorted(root.rglob('*')):
 s=path.lstat(); name=path.relative_to(root).as_posix()
 if stat.S_ISDIR(s.st_mode):
  entries.append([name,'dir',s.st_mode&0o777])
 elif stat.S_ISREG(s.st_mode):
  h=hashlib.sha256()
  with path.open('rb') as source:
   for block in iter(lambda:source.read(1024*1024),b''): h.update(block)
  entries.append([name,'file',s.st_size,h.hexdigest(),s.st_mode&0o777])
 else: raise ValueError('unsafe cache member: '+name)
raw=json.dumps(entries,sort_keys=True,separators=(',',':'),ensure_ascii=True).encode()
print(hashlib.sha256(raw).hexdigest())
'''


class VolumeCache:
    def __init__(self, docker):
        self.docker = docker
        self.sources = {}
        self.records = []

    def directory(self, source, image):
        source = Path(source).absolute()
        def snapshot():
            root = source.lstat()
            if not stat.S_ISDIR(root.st_mode):
                raise ValueError(f'cache source must be a directory: {source}')
            result = []
            for path in sorted(source.rglob('*')):
                s = path.lstat()
                if not (stat.S_ISREG(s.st_mode) or stat.S_ISDIR(s.st_mode)):
                    raise ValueError(f'unsafe cache source: {path}')
                result.append((path.relative_to(source).as_posix(), s.st_dev, s.st_ino, s.st_mode,
                               s.st_size, s.st_mtime_ns, s.st_ctime_ns))
            return result
        started = time.monotonic()
        stamp = snapshot()
        key = ('directory', str(source))
        previous = self.sources.get(key)
        memo_hit = previous is not None and previous[0] == stamp
        if memo_hit:
            digest, size = previous[1:]
        else:
            entries, size = [], 0
            for name, _, _, mode, length, _, _ in stamp:
                if stat.S_ISDIR(mode):
                    entries.append([name, 'dir', mode & 0o777])
                else:
                    entries.append([name, 'file', length, _file_hash(source / name), mode & 0o777])
                    size += length
            if snapshot() != stamp:
                raise DockerError('cache source changed while hashing')
            digest = _digest(entries)
            self.sources[key] = (stamp, digest, size)
        def unchanged():
            if snapshot() != stamp:
                raise DockerError('cache source changed while preparing the volume')
        return self._volume(digest, size, image, [(source, '/cache')], started, memo_hit, unchanged)

    def files(self, sources, image):
        """Cache the public-file allowlist, with its original basenames, without making a temporary data copy."""
        started = time.monotonic()
        entries, uploads, names, stamps = [], [], set(), []
        for item in sorted(sources, key=lambda item: item['name']):
            name, source = item['name'], Path(item['source']).absolute()
            relative = PurePosixPath(name)
            if not relative.parts or relative.name != name or name in ('.', '..') or name in names or source.name != name:
                raise ValueError(f'invalid or duplicate public cache filename: {name}')
            names.add(name)
            before = file_stamp(source)
            key = ('file', str(source))
            previous = self.sources.get(key)
            # build_agent_view has just checked the file against the frozen spec and saved this exact stamp.
            if item.get('verified_stat') == before:
                actual = item['sha256']
            elif previous is not None and previous[0] == before:
                actual = previous[1]
            else:
                actual = _file_hash(source)
            if actual != item['sha256'] or file_stamp(source) != before:
                raise DockerError(f'public cache source changed after validation: {name}')
            self.sources[key] = (before, actual)
            entries.append([name, 'file', before[3], actual, before[2] & 0o777])
            uploads.append((source, '/cache'))
            stamps.append((source, before))
        # Empty allowlists still produce an empty directory, preserving the existing data mount path.
        digest = _digest(entries)
        def unchanged():
            if any(file_stamp(source) != stamp for source, stamp in stamps):
                raise DockerError('public cache sources changed while preparing the volume')
        unchanged()
        return self._volume(digest, sum(entry[2] for entry in entries), image, uploads, started, False, unchanged)

    def _identity(self):
        d = self.docker
        raw = d.info()['raw']
        if not raw.get('ID'):
            raise DockerError('Docker daemon did not report its identity; cannot attest cached volumes')
        context = d.run(['context', 'show']).stdout.strip()
        details = json.loads(d.run(['context', 'inspect', context]).stdout)[0]
        return {'daemon': raw['ID'], 'root': raw.get('DockerRootDir'), 'context': context,
                'endpoint': (details.get('Endpoints') or {}).get('docker'),
                'docker_host': os.environ.get('DOCKER_HOST'), 'docker_context': os.environ.get('DOCKER_CONTEXT')}

    def _inspect(self, name):
        result = self.docker.run(['volume', 'inspect', name], check=False)
        if result.returncode:
            # A daemon failure must not be treated as a missing volume, then overwritten or deleted.
            message = str(result.stderr or '')
            if 'no such volume' not in message.lower():
                raise DockerError(f'cannot inspect cache volume {name}: {message.strip()}')
            return None
        return json.loads(result.stdout)[0]

    def _verify(self, name, image, digest):
        d = self.docker
        cid = d.create(image, ['sleep', 'infinity'], 'cache-' + digest,
                       ['--network', 'none', '--cap-drop', 'ALL', '--security-opt', 'no-new-privileges',
                        '--user', '1000:1000', '--read-only', '--mount',
                        f'type=volume,src={name},dst=/cache,readonly'], {'abench.cache': '1'})
        try:
            d.start(cid)
            actual = d.exec(cid, ['python3', '-c', _VERIFY], timeout=7200).stdout.strip()
            if actual != digest:
                raise DockerError(f'cache volume {name} content does not match its digest')
        except BaseException as exc:
            self._cleanup(cid, primary=exc)
            raise
        else:
            self._cleanup(cid)

    def _cleanup(self, cid, volume=None, primary=None):
        errors = []
        for kind, target, remove in [('container', cid, self.docker.remove),
                                     ('volume', volume, self.docker.remove_volume)]:
            if target is None:
                continue
            try:
                result = remove(target)
                if result is not None and result.returncode:
                    errors.append(f'{kind} {target}: {result.stderr or result.stdout}')
            except BaseException as exc:
                errors.append(f'{kind} {target}: {exc}')
        if errors:
            error = 'cache cleanup failed: ' + '; '.join(errors)
            if primary is not None and hasattr(primary, 'add_note'):
                primary.add_note(error)
            elif primary is None:
                raise DockerError(error)

    def _volume(self, digest, size, image, uploads, started, memo_hit, unchanged):
        d = self.docker
        hash_seconds = time.monotonic() - started
        identity = self._identity()
        cache_root = Path(os.environ.get('ABENCH_CACHE_DIR', str(Path.home() / '.abench/cache'))).expanduser()
        directory = cache_root / 'volumes' / _digest(identity)
        directory.mkdir(parents=True, mode=0o700, exist_ok=True)
        name = 'abench-data-' + digest
        with (directory / (digest + '.lock')).open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            ready = directory / (digest + '.ready.json')
            volume = self._inspect(name)
            hit = volume is not None
            cold = False
            upload_seconds = verification_seconds = 0.0
            if volume is not None:
                labels = volume.get('Labels') or {}
                if labels.get('abench.cache') != '1' or labels.get('abench.cache_digest') != digest:
                    raise DockerError(f'refusing cache volume {name} without matching ABench ownership labels')
                expected = {'schema': 1, 'identity': identity, 'digest': digest, 'volume': volume}
                try:
                    trusted = json.loads(ready.read_text()) == expected
                except (OSError, ValueError):
                    trusted = False
                if not trusted:
                    check_started = time.monotonic()
                    self._verify(name, image, digest)
                    verification_seconds = time.monotonic() - check_started
                    cold = True
            else:
                nonce = uuid.uuid4().hex
                labels = {'abench.cache': '1', 'abench.cache_digest': digest, 'abench.cache_instance': nonce}
                d.volume(name, 'cache-' + digest, labels)
                volume = self._inspect(name)
                if not volume or (volume.get('Labels') or {}).get('abench.cache_instance') != nonce:
                    raise DockerError(f'cache volume {name} was concurrently created by another owner')
                cid = None
                try:
                    cid = d.create(image, ['sleep', 'infinity'], 'cache-' + digest,
                                   ['--network', 'none', '--mount', f'type=volume,src={name},dst=/cache'],
                                   {'abench.cache': '1'})
                    upload_started = time.monotonic()
                    for source, destination in uploads:
                        d.upload(cid, source, destination)
                    upload_seconds = time.monotonic() - upload_started
                    unchanged()
                    # Upload succeeds only after the transport has completed the whole tar stream.
                except BaseException as exc:
                    self._cleanup(cid, volume=name, primary=exc)  # only our nonce-verified new volume
                    raise
                else:
                    self._cleanup(cid)
            unchanged()
            expected = {'schema': 1, 'identity': identity, 'digest': digest, 'volume': volume}
            temporary = ready.with_suffix('.tmp-' + uuid.uuid4().hex)
            try:
                temporary.write_text(_json(expected) + '\n')
                temporary.replace(ready)
            finally:
                temporary.unlink(missing_ok=True)
        self.records.append({'volume': name, 'digest': digest, 'bytes': size, 'source_memo_hit': memo_hit,
                             'hash_seconds': hash_seconds, 'upload_seconds': upload_seconds,
                             'verification_seconds': verification_seconds, 'cold_verified': cold,
                             'hit': hit})
        return name


def for_docker(docker):
    cache = getattr(docker, '_volume_cache', None)
    if cache is None:
        cache = docker._volume_cache = VolumeCache(docker)
        docker.cache_records = cache.records
    return cache
