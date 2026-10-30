"""Fake model service for harness contract tests: Anthropic Messages, OpenAI Responses, OpenAI Chat Completions
and Gemini generateContent, streaming and non-streaming. Replies are the text "pong".

Probes: with probe="shell", a request that offers a shell tool and carries no tool result yet is answered with one
call of that tool running TOOL_COMMAND; the follow-up request (which carries the result) gets "pong". A harness
whose headless approval mode does not really auto-approve then fails to create the probe file. With probe="fetch",
an offered web-fetch tool is called on this server's FETCH_PATH: a request arriving there proves the harness fetches
locally (through the agent's proxy), since a provider-side fetch could never reach the container's loopback.

fail_status answers every model request with that HTTP status and a provider-style error body, to record how each
harness reports authentication, quota, rate-limit and server failures. ForwardProxy is a minimal HTTP proxy
(absolute-form requests and CONNECT) that routes every target to this server; pointing a harness at an unresolvable
host through HTTP(S)_PROXY checks that its model traffic honours the proxy, as it must in an offline container.

It records each request (method, path, credential headers, JSON body) so a test can check that the harness sent
the prompt, the configured model and the credential to the configured base URL. Used in-process by unit tests and
as a script inside agent images (offline, on container loopback):

    python3 fake_model_server.py --port 0 --record /tmp/requests.jsonl   # prints the chosen port
"""
from __future__ import annotations

import argparse
import http.server
import json
import select
import socket
import sys
import threading
import time
from urllib.parse import urlsplit

USAGE = {'input_tokens': 11, 'output_tokens': 2}
TEXT = 'pong'
TOOL_FILE = 'abench_tool_probe.txt'
TOOL_COMMAND = f'echo abench-tool-probe > {TOOL_FILE}'
SHELL_TOOLS = ('Bash', 'bash', 'run_shell_command', 'shell', 'exec_command', 'shell_command', 'Shell', 'execute_command')
FETCH_TOOLS = ('WebFetch', 'web_fetch', 'fetch', 'webfetch')
FETCH_PATH = '/abench-fetch-probe'
ERRORS = {401: ('authentication_error', 'invalid_api_key', 'UNAUTHENTICATED', 'Invalid API key provided (fake).'),
          402: ('billing_error', 'insufficient_quota', 'FAILED_PRECONDITION', 'Insufficient Balance (fake).'),
          404: ('not_found_error', 'model_not_found', 'NOT_FOUND', 'The model `fake-model` does not exist (fake).'),
          429: ('rate_limit_error', 'rate_limit_exceeded', 'RESOURCE_EXHAUSTED', 'Rate limit reached (fake).'),
          500: ('api_error', 'server_error', 'INTERNAL', 'Internal server error (fake).')}


def _probe_call(tools, probe, fetch_url=None, called=()):
    """(name, arguments) of the next probe call, given the offered (name, JSON schema) tools and the tool names the
    conversation already called; None once the probe is done. A fetch tool that is only discoverable (Claude Code
    defers WebFetch behind ToolSearch) is first loaded through ToolSearch."""
    named = {name: schema for name, schema in tools}
    if probe == 'shell' and called:
        return None
    name = next((n for n in (SHELL_TOOLS if probe == 'shell' else FETCH_TOOLS) if n in named), None)
    if probe == 'fetch' and (name is None or name in called):
        if name is None and 'ToolSearch' in named and 'ToolSearch' not in called:
            return 'ToolSearch', {'query': 'select:WebFetch', 'max_results': 1}
        return None
    if name is None:
        return None
    schema = named[name] or {}
    props = schema.get('properties') or {}
    if probe == 'shell':
        key = next((k for k in ('command', 'cmd', 'script') if k in props), 'command')
        args = {key: ['bash', '-lc', TOOL_COMMAND] if (props.get(key) or {}).get('type') == 'array' else TOOL_COMMAND}
    else:
        args = {next((k for k in ('url', 'uri') if k in props), 'url'): fetch_url}
        if 'prompt' in props:
            args['prompt'] = f'Return the page at {fetch_url} verbatim.'
    for other in schema.get('required') or []:          # e.g. DSH's bash requires a description
        kind = (props.get(other) or {}).get('type')
        if other not in args and kind in ('string', 'boolean'):
            args[other] = 'ABench tool probe' if kind == 'string' else False
    return name, args


def error_reply(path, status):
    """A provider-style error body for the protocol the path belongs to."""
    anthropic_type, openai_code, google_status, message = ERRORS.get(status, ('api_error', 'error', 'UNKNOWN', 'Error (fake).'))
    if path.endswith('/messages'):
        body = {'type': 'error', 'error': {'type': anthropic_type, 'message': message}}
    elif ':generateContent' in path or ':streamGenerateContent' in path:
        body = {'error': {'code': status, 'message': message, 'status': google_status}}
    else:
        body = {'error': {'message': message, 'type': openai_code, 'code': openai_code, 'param': None}}
    return json.dumps(body).encode()


def _sse(events, named=True):
    out = []
    for name, data in events:
        out.append((f'event: {name}\n' if named and name else '') + 'data: ' + json.dumps(data) + '\n\n')
    return ''.join(out).encode()


def anthropic(body, probe=None, fetch_url=None):
    model = body.get('model', 'fake-model')
    called = [b.get('name') for m in body.get('messages') or [] if m.get('role') == 'assistant'
              for b in (m.get('content') if isinstance(m.get('content'), list) else []) if isinstance(b, dict) and b.get('type') == 'tool_use']
    call = None if not probe else _probe_call([(t.get('name'), t.get('input_schema')) for t in body.get('tools') or []], probe, fetch_url, called)
    if call:
        block = {'type': 'tool_use', 'id': f'toolu_fake{len(called)}', 'name': call[0], 'input': call[1]}
        message = {'id': 'msg_fake', 'type': 'message', 'role': 'assistant', 'model': model, 'content': [block],
                   'stop_reason': 'tool_use', 'stop_sequence': None, 'usage': USAGE}
        if not body.get('stream'):
            return 'application/json', json.dumps(message).encode()
        return 'text/event-stream', _sse([
            ('message_start', {'type': 'message_start', 'message': {**message, 'content': [], 'stop_reason': None}}),
            ('content_block_start', {'type': 'content_block_start', 'index': 0, 'content_block': {**block, 'input': {}}}),
            ('content_block_delta', {'type': 'content_block_delta', 'index': 0,
                                     'delta': {'type': 'input_json_delta', 'partial_json': json.dumps(call[1])}}),
            ('content_block_stop', {'type': 'content_block_stop', 'index': 0}),
            ('message_delta', {'type': 'message_delta', 'delta': {'stop_reason': 'tool_use', 'stop_sequence': None}, 'usage': USAGE}),
            ('message_stop', {'type': 'message_stop'})])
    message = {'id': 'msg_fake', 'type': 'message', 'role': 'assistant', 'model': model,
               'content': [{'type': 'text', 'text': TEXT}], 'stop_reason': 'end_turn', 'stop_sequence': None,
               'usage': USAGE}
    if not body.get('stream'):
        return 'application/json', json.dumps(message).encode()
    return 'text/event-stream', _sse([
        ('message_start', {'type': 'message_start', 'message': {**message, 'content': [], 'stop_reason': None}}),
        ('content_block_start', {'type': 'content_block_start', 'index': 0, 'content_block': {'type': 'text', 'text': ''}}),
        ('content_block_delta', {'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': TEXT}}),
        ('content_block_stop', {'type': 'content_block_stop', 'index': 0}),
        ('message_delta', {'type': 'message_delta', 'delta': {'stop_reason': 'end_turn', 'stop_sequence': None}, 'usage': USAGE}),
        ('message_stop', {'type': 'message_stop'})])


def responses(body, probe=None, fetch_url=None):
    model = body.get('model', 'fake-model')
    items = body.get('input') if isinstance(body.get('input'), list) else []
    called = [i.get('name') for i in items if isinstance(i, dict) and i.get('type') == 'function_call']
    tools = [(t.get('name'), t.get('parameters')) for t in body.get('tools') or [] if t.get('type') == 'function']
    call = None if not probe else _probe_call(tools, probe, fetch_url, called)
    if call:
        item = {'id': f'fc_fake{len(called)}', 'type': 'function_call', 'status': 'completed', 'call_id': f'call_fake{len(called)}',
                'name': call[0], 'arguments': json.dumps(call[1])}
    else:
        item = {'id': 'msg_fake', 'type': 'message', 'status': 'completed', 'role': 'assistant',
                'content': [{'type': 'output_text', 'text': TEXT, 'annotations': []}]}
    usage = {'input_tokens': USAGE['input_tokens'], 'output_tokens': USAGE['output_tokens'],
             'total_tokens': USAGE['input_tokens'] + USAGE['output_tokens'],
             'input_tokens_details': {'cached_tokens': 0}, 'output_tokens_details': {'reasoning_tokens': 0}}
    response = {'id': 'resp_fake', 'object': 'response', 'created_at': int(time.time()), 'status': 'completed',
                'model': model, 'output': [item], 'usage': usage}
    if not body.get('stream'):
        return 'application/json', json.dumps(response).encode()
    pending = {**response, 'status': 'in_progress', 'output': [], 'usage': None}
    if item['type'] == 'function_call':
        events = [('response.created', {'type': 'response.created', 'response': pending}),
                  ('response.output_item.added', {'type': 'response.output_item.added', 'output_index': 0,
                                                  'item': {**item, 'status': 'in_progress', 'arguments': ''}}),
                  ('response.function_call_arguments.delta', {'type': 'response.function_call_arguments.delta',
                                                             'item_id': item['id'], 'output_index': 0, 'delta': item['arguments']}),
                  ('response.function_call_arguments.done', {'type': 'response.function_call_arguments.done',
                                                            'item_id': item['id'], 'output_index': 0, 'arguments': item['arguments']}),
                  ('response.output_item.done', {'type': 'response.output_item.done', 'output_index': 0, 'item': item}),
                  ('response.completed', {'type': 'response.completed', 'response': response})]
        return 'text/event-stream', _sse([(n, {**d, 'sequence_number': i}) for i, (n, d) in enumerate(events)])
    part = {'type': 'output_text', 'text': '', 'annotations': []}
    events = [('response.created', {'type': 'response.created', 'response': pending}),
              ('response.in_progress', {'type': 'response.in_progress', 'response': pending}),
              ('response.output_item.added', {'type': 'response.output_item.added', 'output_index': 0,
                                              'item': {**item, 'status': 'in_progress', 'content': []}}),
              ('response.content_part.added', {'type': 'response.content_part.added', 'item_id': 'msg_fake',
                                               'output_index': 0, 'content_index': 0, 'part': part}),
              ('response.output_text.delta', {'type': 'response.output_text.delta', 'item_id': 'msg_fake',
                                              'output_index': 0, 'content_index': 0, 'delta': TEXT}),
              ('response.output_text.done', {'type': 'response.output_text.done', 'item_id': 'msg_fake',
                                             'output_index': 0, 'content_index': 0, 'text': TEXT}),
              ('response.content_part.done', {'type': 'response.content_part.done', 'item_id': 'msg_fake',
                                              'output_index': 0, 'content_index': 0, 'part': {**part, 'text': TEXT}}),
              ('response.output_item.done', {'type': 'response.output_item.done', 'output_index': 0, 'item': item}),
              ('response.completed', {'type': 'response.completed', 'response': response})]
    return 'text/event-stream', _sse([(n, {**d, 'sequence_number': i}) for i, (n, d) in enumerate(events)])


def chat(body, probe=None, fetch_url=None):
    model = body.get('model', 'fake-model')
    called = [(c.get('function') or {}).get('name') for m in body.get('messages') or [] if m.get('role') == 'assistant'
              for c in m.get('tool_calls') or []]
    tools = [((t.get('function') or {}).get('name'), (t.get('function') or {}).get('parameters')) for t in body.get('tools') or []]
    call = None if not probe else _probe_call(tools, probe, fetch_url, called)
    usage = {'prompt_tokens': USAGE['input_tokens'], 'completion_tokens': USAGE['output_tokens'],
             'total_tokens': USAGE['input_tokens'] + USAGE['output_tokens']}
    base = {'id': 'chatcmpl-fake', 'created': int(time.time()), 'model': model}
    if call:
        tool_call = {'id': f'call_fake{len(called)}', 'type': 'function', 'function': {'name': call[0], 'arguments': json.dumps(call[1])}}
        if not body.get('stream'):
            return 'application/json', json.dumps({**base, 'object': 'chat.completion', 'usage': usage, 'choices': [
                {'index': 0, 'finish_reason': 'tool_calls',
                 'message': {'role': 'assistant', 'content': None, 'tool_calls': [tool_call]}}]}).encode()
        chunk = {**base, 'object': 'chat.completion.chunk'}
        first = {'index': 0, 'id': tool_call['id'], 'type': 'function', 'function': {'name': call[0], 'arguments': ''}}
        events = [(None, {**chunk, 'choices': [{'index': 0, 'delta': {'role': 'assistant', 'content': None, 'tool_calls': [first]}, 'finish_reason': None}]}),
                  (None, {**chunk, 'choices': [{'index': 0, 'delta': {'tool_calls': [{'index': 0, 'function': {'arguments': json.dumps(call[1])}}]}, 'finish_reason': None}]}),
                  (None, {**chunk, 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'tool_calls'}]}),
                  (None, {**chunk, 'choices': [], 'usage': usage})]
        return 'text/event-stream', _sse(events, named=False) + b'data: [DONE]\n\n'
    if not body.get('stream'):
        return 'application/json', json.dumps({**base, 'object': 'chat.completion', 'usage': usage, 'choices': [
            {'index': 0, 'finish_reason': 'stop', 'message': {'role': 'assistant', 'content': TEXT}}]}).encode()
    chunk = {**base, 'object': 'chat.completion.chunk'}
    events = [(None, {**chunk, 'choices': [{'index': 0, 'delta': {'role': 'assistant', 'content': ''}, 'finish_reason': None}]}),
              (None, {**chunk, 'choices': [{'index': 0, 'delta': {'content': TEXT}, 'finish_reason': None}]}),
              (None, {**chunk, 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]}),
              (None, {**chunk, 'choices': [], 'usage': usage})]
    return 'text/event-stream', _sse(events, named=False) + b'data: [DONE]\n\n'


def gemini(body, model, stream, probe=None, fetch_url=None):
    called = [part['functionCall'].get('name') for c in body.get('contents') or [] for part in c.get('parts') or []
              if 'functionCall' in part]
    tools = [(d.get('name'), d.get('parametersJsonSchema') or d.get('parameters'))
             for t in body.get('tools') or [] for d in t.get('functionDeclarations') or []]
    call = None if not probe else _probe_call(tools, probe, fetch_url, called)
    if call:
        reply = {'candidates': [{'content': {'role': 'model', 'parts': [{'functionCall': {'name': call[0], 'args': call[1]}}]},
                                 'finishReason': 'STOP', 'index': 0}],
                 'usageMetadata': {'promptTokenCount': USAGE['input_tokens'], 'candidatesTokenCount': USAGE['output_tokens'],
                                   'totalTokenCount': USAGE['input_tokens'] + USAGE['output_tokens']},
                 'modelVersion': model, 'responseId': 'gemini-fake'}
        if not stream:
            return 'application/json', json.dumps(reply).encode()
        return 'text/event-stream', _sse([(None, reply)], named=False)
    reply = {'candidates': [{'content': {'role': 'model', 'parts': [{'text': TEXT}]}, 'finishReason': 'STOP', 'index': 0}],
             'usageMetadata': {'promptTokenCount': USAGE['input_tokens'], 'candidatesTokenCount': USAGE['output_tokens'],
                               'totalTokenCount': USAGE['input_tokens'] + USAGE['output_tokens']},
             'modelVersion': model, 'responseId': 'gemini-fake'}
    if not stream:
        return 'application/json', json.dumps(reply).encode()
    return 'text/event-stream', _sse([(None, reply)], named=False)


def make_handler(record, probe=None, fail_status=None, fetch_url=None):
    class Handler(http.server.BaseHTTPRequestHandler):
        protocol_version = 'HTTP/1.1'

        def log_message(self, *args):
            pass

        def _reply(self, status, content_type, data):
            self.send_response(status)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _note(self, body):
            url = urlsplit(self.path)
            entry = {'method': self.command, 'path': url.path, 'query': url.query,
                     'authorization': self.headers.get('Authorization'), 'x-api-key': self.headers.get('x-api-key'),
                     'x-goog-api-key': self.headers.get('x-goog-api-key'), 'body': body}
            record(entry)
            return url.path

        def do_GET(self):
            path = self._note(None)
            if path == FETCH_PATH:
                return self._reply(200, 'text/plain', b'abench-fetch-probe-content\n')
            if path.endswith('/models'):
                ids = ['fake-model']
                data = {'object': 'list', 'data': [{'id': i, 'object': 'model', 'owned_by': 'abench'} for i in ids],
                        'models': [{'name': 'models/' + i, 'displayName': i, 'slug': i} for i in ids]}
                return self._reply(200, 'application/json', json.dumps(data).encode())
            self._reply(404, 'application/json', b'{"error":{"message":"not found"}}')

        def do_POST(self):
            raw = self.rfile.read(int(self.headers.get('Content-Length') or 0))
            try:
                body = json.loads(raw or b'{}')
            except ValueError:
                body = {'_raw': raw.decode('utf-8', 'replace')}
            path = self._note(body)
            if fail_status and not path.endswith('/count_tokens'):
                return self._reply(fail_status, 'application/json', error_reply(path, fail_status))
            probe_url = fetch_url or f'http://127.0.0.1:{self.server.server_port}{FETCH_PATH}'
            if path.endswith('/messages/count_tokens'):
                return self._reply(200, 'application/json', b'{"input_tokens": 11}')
            if path.endswith('/messages'):
                return self._reply(200, *anthropic(body, probe, probe_url))
            if path.endswith('/responses'):
                return self._reply(200, *responses(body, probe, probe_url))
            if path.endswith('/chat/completions'):
                return self._reply(200, *chat(body, probe, probe_url))
            if ':generateContent' in path or ':streamGenerateContent' in path:
                model = path.rsplit('/', 1)[-1].split(':', 1)[0]
                return self._reply(200, *gemini(body, model, ':streamGenerateContent' in path, probe, probe_url))
            if ':countTokens' in path:
                return self._reply(200, 'application/json', b'{"totalTokens": 11}')
            self._reply(404, 'application/json', b'{"error":{"message":"not found"}}')
    return Handler


class ForwardProxy:
    """HTTP proxy on 127.0.0.1 that routes every absolute-form request and CONNECT tunnel to one local port and
    records the requested target, so a test can see which traffic used the proxy."""
    def __init__(self, target_port, record, port=0):
        self.target_port, self.record = target_port, record
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(('127.0.0.1', port))
        self.sock.listen(64)
        self.port = self.sock.getsockname()[1]
        self.thread = threading.Thread(target=self._serve, daemon=True)

    def start(self):
        self.thread.start()
        return self

    def close(self):
        self.sock.close()

    def _serve(self):
        while True:
            try:
                client, _ = self.sock.accept()
            except OSError:
                return
            threading.Thread(target=self._handle, args=(client,), daemon=True).start()

    def _handle(self, client):
        try:
            head = b''
            while b'\r\n\r\n' not in head:
                chunk = client.recv(65536)
                if not chunk:
                    return
                head += chunk
            line, _, rest = head.partition(b'\r\n')
            method, target, version = line.decode('latin-1').split(' ', 2)
            upstream = socket.create_connection(('127.0.0.1', self.target_port))
            if method == 'CONNECT':
                self.record({'proxy': True, 'method': method, 'target': target})
                client.sendall(b'HTTP/1.1 200 Connection Established\r\n\r\n')
                leftover = head.split(b'\r\n\r\n', 1)[1]
            else:
                url = urlsplit(target)
                self.record({'proxy': True, 'method': method, 'target': url.netloc})
                path = (url.path or '/') + ('?' + url.query if url.query else '')
                leftover = f'{method} {path} {version}\r\n'.encode() + rest
            if leftover:
                upstream.sendall(leftover)
            sockets = [client, upstream]
            while True:
                ready, _, _ = select.select(sockets, [], [], 120)
                if not ready:
                    return
                for s in ready:
                    data = s.recv(65536)
                    if not data:
                        return
                    (upstream if s is client else client).sendall(data)
        except (OSError, ValueError):
            pass
        finally:
            client.close()


class FakeModelServer:
    """In-process server on 127.0.0.1; requests accumulate in .requests. With proxy=True, .proxy_url is a
    ForwardProxy in front of it."""
    def __init__(self, probe=None, fail_status=None, proxy=False, fetch_url=None):
        self.requests = []
        self.lock = threading.Lock()
        def record(entry):
            with self.lock:
                self.requests.append(entry)
        self.server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), make_handler(record, probe, fail_status, fetch_url))
        self.port = self.server.server_port
        self.url = f'http://127.0.0.1:{self.port}'
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.proxy = ForwardProxy(self.port, record) if proxy else None
        self.proxy_url = f'http://127.0.0.1:{self.proxy.port}' if proxy else None

    def __enter__(self):
        self.thread.start()
        if self.proxy:
            self.proxy.start()
        return self

    def __exit__(self, *exc):
        if self.proxy:
            self.proxy.close()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--port', type=int, default=0)
    ap.add_argument('--record', required=True, help='JSONL file receiving one line per request')
    ap.add_argument('--probe', choices=('shell', 'fetch'), help='answer the first shell or web-fetch tool offer with a call')
    ap.add_argument('--fail-status', type=int, help='answer every model request with this HTTP error status')
    ap.add_argument('--proxy-port', type=int, help='also run a ForwardProxy on this port (0: any)')
    a = ap.parse_args()
    lock = threading.Lock()
    def record(entry):
        with lock, open(a.record, 'a') as f:
            f.write(json.dumps(entry) + '\n')
    server = http.server.ThreadingHTTPServer(('127.0.0.1', a.port), make_handler(record, a.probe, a.fail_status))
    proxy = ForwardProxy(server.server_port, record, a.proxy_port).start() if a.proxy_port is not None else None
    print(server.server_port, *([proxy.port] if proxy else []), flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == '__main__':
    sys.exit(main())
