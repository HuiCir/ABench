"""Classify why a harness run failed, from the harness's own structured events and diagnostics (never tool output).

A provider error (bad key, exhausted quota, rate limit, provider outage, unknown model) means the run did not
measure the agent: it is unscorable, and jobs retry the transient kinds instead of counting a failure. Every other
failure (agent_error, timeout, context_length) is the agent's own result."""
from __future__ import annotations

import re

PROVIDER_ERRORS = ('provider_auth', 'provider_quota', 'provider_rate_limit', 'provider_unavailable', 'model_not_found',
                   'provider_rejected')          # provider_rejected: a preflight request refused for another reason
RETRYABLE = ('provider_rate_limit', 'provider_unavailable')

# keyword groups, checked in order: quota before rate limit (OpenAI reports insufficient_quota with status 429)
_KEYWORDS = (
    ('provider_quota', ('insufficient balance', 'insufficient_quota', 'quota', 'billing', 'payment required', 'credit balance')),
    ('provider_auth', ('invalid api key', 'invalid_api_key', 'unauthorized', 'unauthenticated', 'authentication',
                       'auth_failed', 'api key not valid', 'permission denied', 'permission_denied')),
    ('provider_rate_limit', ('rate limit', 'rate_limit', 'too many requests', 'resource_exhausted', 'rate_limited')),
    ('model_not_found', ('model_not_found', 'model not found', 'unknown model', 'does not exist')),
    ('context_length', ('context length', 'context_length', 'context window', 'maximum context', 'prompt is too long',
                        'too many tokens')),
    ('provider_unavailable', ('overloaded', 'server error', 'server_error', 'internal error', 'service unavailable',
                              'bad gateway', 'gateway timeout', 'high demand', 'temporarily unavailable',
                              'connection refused', 'connection reset', 'econnrefused', 'enotfound', 'fetch failed',
                              'network error', 'timed out')),
)
_STATUS = re.compile(r'(?:status|code|error|api error)\D{0,4}([45]\d\d)\b', re.IGNORECASE)


def status_of(text: str) -> int | None:
    match = _STATUS.search(text or '')
    return int(match[1]) if match else None


def classify(status: int | None = None, text: str = '') -> str | None:
    """error_type of one failure from its HTTP status and message; None when neither identifies it."""
    lowered = (text or '').lower()
    status = status or status_of(lowered)
    if status == 402:
        return 'provider_quota'
    for kind, words in _KEYWORDS:
        if any(w in lowered for w in words):
            return kind
    if status in (401, 403):
        return 'provider_auth'
    if status == 429:
        return 'provider_rate_limit'
    if status == 404:
        return 'model_not_found'
    if status and status >= 500:
        return 'provider_unavailable'
    return None


def run_error_type(status: str, summary: dict | None) -> str | None:
    """Why a run failed; None when it succeeded. A provider error counts only when it ended the run: the harness's
    last event, or what it was still retrying when the session deadline arrived."""
    summary = summary or {}
    result = summary.get('result') or {}
    if status == 'succeeded' and not result.get('is_error'):
        return None
    provider = summary.get('provider_error') or {}
    if provider.get('terminal') and provider.get('error_type'):
        return provider['error_type']
    if status == 'timeout':
        return 'timeout'
    if classify(None, str(result.get('error') or result.get('final_text') or '')) == 'context_length':
        return 'context_length'
    return 'agent_error'


# Why judging a collected submission stopped, by exception class (the first match along its MRO), in the manner of a
# central policy table: Docker, transfer and connection failures are the infrastructure's, so the same submission is
# judged again without rerunning the agent; anything else (a changed case, a rejected container configuration, a
# hash mismatch) is final.
JUDGE_FAILURES = {'DockerError': ('judge_infrastructure', True), 'TimeoutExpired': ('judge_infrastructure', True),
                  'TimeoutError': ('judge_infrastructure', True), 'ConnectionError': ('judge_infrastructure', True)}


def judge_failure(exc: BaseException) -> dict:
    """phase, exception_type, kind and retryable of an exception that stopped judging."""
    kind, retryable = next((JUDGE_FAILURES[c.__name__] for c in type(exc).__mro__ if c.__name__ in JUDGE_FAILURES),
                           ('judge_error', False))
    return {'phase': 'judge', 'exception_type': type(exc).__name__, 'kind': kind, 'retryable': retryable}


def retryable_judge_failure(evaluation: dict | None) -> bool:
    """A judge_error evaluation caused by the infrastructure: the collected submission can be judged again."""
    evaluation = evaluation or {}
    failure = evaluation.get('failure') or {}
    return evaluation.get('verdict') == 'judge_error' and failure.get('phase') == 'judge' and failure.get('retryable') is True
