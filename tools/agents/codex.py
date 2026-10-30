import json
from .base import Agent, Launch, pinned


def _toml(value):
    return json.dumps(value)        # a JSON string literal is a valid TOML basic string


class Codex(Agent):
    """OpenAI Codex CLI (`codex exec --json`). Codex speaks only the Responses API; the endpoint becomes a custom
    model provider in an isolated CODEX_HOME, and the key stays in the endpoint's own environment variable."""
    name = "codex"
    version = pinned("codex")["version"]
    protocols = ("openai-responses",)
    default_model = "gpt-6-astra"
    needs_network = True
    parameters = ("effort",)
    disabled_tools = ("web_search",)        # an OpenAI-hosted tool; Codex has no local fetch tool
    # Codex passes any value to the API unchecked, so a typo would only fail mid-run.
    efforts = ("none", "minimal", "low", "medium", "high", "xhigh", "max")
    session_logs = (".codex/sessions",)
    cli_version_argv = ("codex", "--version")
    cli_version_prefix = "codex-cli "

    def launch(self, args, paths, prompt, endpoint=None):
        options = self.options(endpoint)
        lines = [f"model = {_toml(endpoint['model'])}", 'model_provider = "abench"',
                 "check_for_update_on_startup = false", 'web_search = "disabled"']
        if options.get("effort"):
            lines.append(f"model_reasoning_effort = {_toml(options['effort'])}")
        lines += ["", "[analytics]", "enabled = false", "", "[model_providers.abench]", 'name = "ABench model endpoint"',
                  f"base_url = {_toml(endpoint['base_url'])}", f"env_key = {_toml(endpoint['key_env'])}",
                  'wire_api = "responses"']
        argv = ["codex", "exec", "--json", "--strict-config", "--skip-git-repo-check",
                # the container is the sandbox; Codex's own sandbox needs kernel features containers withhold
                "--dangerously-bypass-approvals-and-sandbox", "-C", str(paths["workspace"]), "-"]
        return Launch(argv, {"CODEX_HOME": str(paths["home"] / ".codex")}, prompt.encode(),
                      {".codex/config.toml": "\n".join(lines) + "\n"})

    def summarize(self, raw, stderr=b""):
        from .trajectory import summarize_codex
        return summarize_codex(raw, stderr)
