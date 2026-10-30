"""One tiny authenticated request per model protocol, sent from the agent container before the timed session.

It uses the same base URL, credential variable and proxy path as the harness, so an unreachable endpoint or a
rejected key stops the run before any development time or budget is spent."""
import json

from .base import Launch

CODE = r'''import json, os, urllib.error, urllib.request
api, base, model = os.environ['ABENCH_PREFLIGHT_API'], os.environ['ABENCH_PREFLIGHT_URL'].rstrip('/'), os.environ['ABENCH_PREFLIGHT_MODEL']
key = os.environ[os.environ['ABENCH_PREFLIGHT_KEY_ENV']]
ask = 'Reply pong'
if api == 'anthropic':
    url, headers = base + '/v1/messages', {'x-api-key': key, 'anthropic-version': '2023-06-01'}
    body = {'model': model, 'max_tokens': 16, 'messages': [{'role': 'user', 'content': ask}]}
elif api == 'openai-responses':
    url, headers = base + '/responses', {'Authorization': 'Bearer ' + key}
    body = {'model': model, 'input': ask, 'max_output_tokens': 64}
elif api == 'openai-chat':
    url, headers = base + '/chat/completions', {'Authorization': 'Bearer ' + key}
    body = {'model': model, 'max_tokens': 16, 'messages': [{'role': 'user', 'content': ask}]}
elif api == 'gemini':
    url, headers = base + '/v1beta/models/' + model + ':generateContent', {'x-goog-api-key': key}
    body = {'contents': [{'role': 'user', 'parts': [{'text': ask}]}], 'generationConfig': {'maxOutputTokens': 64}}
else:
    raise SystemExit('unsupported protocol ' + api)
request = urllib.request.Request(url, data=json.dumps(body).encode(), headers={'Content-Type': 'application/json', **headers})
try:
    with urllib.request.urlopen(request, timeout=45) as response:
        reply = json.load(response)
except urllib.error.HTTPError as error:
    print(json.dumps({'abench_preflight_error': {'status': error.code, 'message': error.read(2000).decode('utf-8', 'replace')}}))
    raise SystemExit(3)
except (urllib.error.URLError, OSError, ValueError) as error:
    print(json.dumps({'abench_preflight_error': {'status': None, 'message': 'connection error: ' + str(getattr(error, 'reason', error))}}))
    raise SystemExit(3)
if not isinstance(reply, dict) or 'error' in reply:
    raise SystemExit('model preflight returned no usable response')
print('authenticated')
'''


def preflight_failure(output: bytes) -> dict:
    """Classified failure of a preflight run from its output; a connection failure counts as provider_unavailable."""
    from .errors import classify
    for line in reversed(output.decode('utf-8', 'replace').splitlines()):
        if '"abench_preflight_error"' in line:
            try:
                error = json.loads(line)['abench_preflight_error']
            except (ValueError, KeyError):
                break
            kind = classify(error.get('status'), error.get('message', '')) or (
                'provider_unavailable' if error.get('status') is None else 'provider_rejected')
            return {'error_type': kind, 'status': error.get('status'), 'message': str(error.get('message', ''))[:500]}
    return {'error_type': 'provider_unavailable', 'status': None, 'message': 'preflight produced no response'}


def protocol_preflight(endpoint) -> Launch:
    return Launch(['python3', '-c', CODE], {'ABENCH_PREFLIGHT_API': endpoint['api'],
                                            'ABENCH_PREFLIGHT_URL': endpoint['base_url'],
                                            'ABENCH_PREFLIGHT_MODEL': endpoint['model'],
                                            'ABENCH_PREFLIGHT_KEY_ENV': endpoint['key_env']})
