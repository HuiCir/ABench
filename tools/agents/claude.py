import json
from .base import Agent, Launch, pinned

class Claude(Agent):
    name = "claude"
    version = pinned("claude")["version"]
    protocols = ("anthropic",)
    default_model = "claude-opus-5-5"
    needs_network = True
    parameters = ("max_output_tokens", "effort")
    # WebSearch runs at Anthropic. WebFetch fetches locally through the proxy once its claude.ai domain lookup is
    # skipped (that lookup needs api.anthropic.com, which other endpoints' egress policies deny).
    disabled_tools = ("WebSearch",)
    efforts = ("low", "medium", "high", "xhigh", "max")         # `claude --help`
    session_logs = (".claude/projects",)
    cli_version_argv = ("claude", "--version")
    cli_version_suffix = " (Claude Code)"
    settings = {"skipWebFetchPreflight": True, "permissions": {"deny": list(disabled_tools)}}

    def version_probe(self, args):
        return Launch([getattr(args, "claude_bin", "claude"), "--version"], self.environment(None))

    def credential_env(self, endpoint):
        return ("ANTHROPIC_API_KEY",)

    def environment(self, endpoint):
        env = {"DISABLE_AUTOUPDATER": "1", "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
               "DISABLE_TELEMETRY": "1", "DISABLE_ERROR_REPORTING": "1"}
        if endpoint:
            env.update(ANTHROPIC_BASE_URL=endpoint["base_url"], ANTHROPIC_MODEL=endpoint["model"])
            aux = endpoint.get("auxiliary_models", {})
            for name, value in aux.items():
                if name not in ("small", "haiku", "sonnet", "opus"):
                    raise ValueError(f"unsupported Claude auxiliary model: {name}")
                env[{"small": "ANTHROPIC_SMALL_FAST_MODEL", "haiku": "ANTHROPIC_DEFAULT_HAIKU_MODEL",
                     "sonnet": "ANTHROPIC_DEFAULT_SONNET_MODEL", "opus": "ANTHROPIC_DEFAULT_OPUS_MODEL"}[name]] = value
        return env

    def launch(self, args, paths, prompt, endpoint=None):
        options = self.options(endpoint)
        env = self.environment(endpoint)
        if options.get("max_output_tokens"):
            env["CLAUDE_CODE_MAX_OUTPUT_TOKENS"] = str(options["max_output_tokens"])
        # Sessions persist under HOME so the transcript can be kept as evidence; HOME is per run.
        cmd = [getattr(args, "claude_bin", "claude"), "-p", "--output-format", "stream-json", "--verbose"]
        model = (endpoint or {}).get("model") or getattr(args, "claude_model", None)
        if model:
            cmd += ["--model", model]
        effort = getattr(args, "claude_effort", None) or options.get("effort")
        self.check_effort(endpoint, effort)
        if effort:
            cmd += ["--effort", effort]
        cap = getattr(args, "claude_max_budget_usd", None)
        if cap is not None:
            cmd += ["--max-budget-usd", str(cap)]
        if getattr(args, "backend", "host") == "docker":
            cmd += ["--permission-mode", "bypassPermissions", "--dangerously-skip-permissions"]
        else:
            cmd += ["--permission-mode", "acceptEdits", "--allowedTools", "Bash,Read,Write,Edit,Glob,Grep"]
        cmd += ["--disallowedTools", ",".join(self.disabled_tools)]
        cmd += ["--add-dir", str(paths["workspace"]), "--add-dir", str(paths["view"])]
        return Launch(cmd, env, prompt.encode(), {".claude/settings.json": json.dumps(self.settings, indent=2) + "\n"})

    def summarize(self, raw, stderr=b""):
        from sandbox_runner import summarize_stream
        from .trajectory import claude_to_atif, provider_error
        summary = summarize_stream(raw)
        if summary:
            models, usage, provider = set(), {}, None
            for line in raw.splitlines():
                try:
                    e = json.loads(line)
                except (ValueError, TypeError):
                    continue
                if not isinstance(e, dict):
                    continue
                if provider:                      # a later event means the provider error did not end the run
                    provider["terminal"] = False
                m = e.get("message") or {}
                if m.get("model") and not str(m["model"]).startswith("<"):
                    models.add(m["model"])
                if e.get("type") == "system" and e.get("subtype") == "api_retry":
                    provider = provider_error(e.get("error_status"), e.get("error"), terminal=True)
                if e.get("type") == "result":
                    usage = e.get("usage") or {}
                    if e.get("is_error") and (e.get("api_error_status") or str(e.get("result", "")).startswith("API Error")):
                        provider = provider_error(e.get("api_error_status"), e.get("result"), terminal=True)
            summary.update(actual_models=sorted(models), usage=usage, provider_error=provider,
                           atif=claude_to_atif(raw, self.version))
        return summary
