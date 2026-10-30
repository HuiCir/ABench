#!/usr/bin/env python3
"""Write a harness configuration that must contain the model key, then exec the harness.

Some CLIs (ZCode) read the API key only from their configuration file. The adapter writes the configuration
without the key; inside the agent container this helper copies it to a private (0600) file with the key taken from
the named environment variable, so the key never appears in argv, in runner-written files or in the run record.

    python3 secret_config.py --template T --out O --key-env NAME --placeholder TEXT -- harness args...
"""
import argparse
import os
import sys


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--template', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--key-env', required=True)
    ap.add_argument('--placeholder', required=True)
    ap.add_argument('command', nargs=argparse.REMAINDER)
    a = ap.parse_args(argv)
    command = a.command[1:] if a.command[:1] == ['--'] else a.command
    if not command:
        ap.error('missing harness command')
    key = os.environ.get(a.key_env)
    if not key:
        ap.error(f'{a.key_env} is not set')
    with open(a.template, encoding='utf-8') as f:
        text = f.read()
    if text.count(a.placeholder) != 1:
        ap.error('the template must contain the placeholder exactly once')
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    fd = os.open(a.out, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, 'w', encoding='utf-8') as f:
        f.write(text.replace(a.placeholder, key.replace('\\', '\\\\').replace('"', '\\"')))
    os.execvp(command[0], command)


if __name__ == '__main__':
    sys.exit(main())
