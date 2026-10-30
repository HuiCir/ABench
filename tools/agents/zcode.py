import json
import re
from .base import Agent, Launch, pinned

PROTOCOL = {"openai-chat": "openai-chat-completions", "openai-responses": "openai-responses",
            "anthropic": "anthropic-messages"}
PLACEHOLDER = "ABENCH_KEY_PLACEHOLDER"
PROMPT_LIMIT = 120_000          # the prompt is one argv string; Linux caps a single argument at 128 KiB


class ZCode(Agent):
    """Z.ai ZCode (`zcode --prompt ... --mode yolo --output-format stream-json`). ZCode reads the key only from its
    provider file, so the adapter writes a keyless template and secret_config.py fills it inside the container."""
    name = "zcode"
    version = pinned("zcode")["version"]
    protocols = tuple(PROTOCOL)
    default_model = "glm-5.3"
    needs_network = True
    parameters = ("effort",)
    disabled_tools = ("WebSearch",)         # provider-native search for some protocols; WebFetch is local
    efforts = ("disabled", "enabled", "low", "high", "max")
    session_logs = (".zcode/cli/rollout",)

    def effort_values(self, endpoint):
        # ZCode's built-in model rules (v3.15.1) give each model family its own reasoning levels.
        model = str(endpoint.get("model", "")).lower()
        if re.search(r"glm-5\.3", model):
            return ("low", "high", "max")
        if re.search(r"glm-5\.2", model):
            return ("disabled", "high", "max")
        return ("disabled", "enabled")
    cli_version_argv = ("zcode", "--version")

    def proxy_env(self, proxy_url):
        # ZCode ignores HTTP(S)_PROXY for model and fetch traffic; it has its own variables.
        return {"ZCODE_HTTP_PROXY": proxy_url, "ZCODE_NO_PROXY": "localhost,127.0.0.1"}

    def launch(self, args, paths, prompt, endpoint=None):
        options = self.options(endpoint)
        if len(prompt.encode()) > PROMPT_LIMIT:
            raise ValueError("prompt exceeds the single-argument limit of the zcode CLI")
        model = endpoint["model"]
        selection = {"providerId": "abench", "modelId": model}
        if options.get("effort"):
            selection["options"] = {"reasoningLevel": options["effort"]}
        config = {"schemaVersion": 1, "config": {
            "providerConfigRules": {"providerRules": [{"providerId": "abench", "providerName": "ABench model endpoint", "config": {
                "group": "standard-personal", "access": {"type": "api-key", "apiKey": PLACEHOLDER},
                "api": {"type": PROTOCOL[endpoint["api"]], "baseUrl": endpoint["base_url"]},
                "personalModelIds": [model], "modelOrder": [model]}}]},
            "modelConfigRules": {"providerModelRules": [], "manualProviderModelRules": []},
            "defaultModelSelection": selection}}
        home = paths["home"]
        argv = ["python3", "/abench/control/secret_config.py", "--template", str(home / ".zcode-abench/provider_config.template.json"),
                "--out", str(home / ".zcode-abench/provider_config.json"), "--key-env", endpoint["key_env"],
                "--placeholder", PLACEHOLDER, "--",
                "zcode", "--prompt", prompt, "--cwd", str(paths["workspace"]), "--mode", "yolo", "--output-format", "stream-json",
                "--disallowed-tools", ",".join(self.disabled_tools)]
        env = {"ZCODE_PERSONAL_PROVIDER_CONFIG_FILE": str(home / ".zcode-abench/provider_config.json")}
        return Launch(argv, env, b"", {".zcode-abench/provider_config.template.json": json.dumps(config, indent=2) + "\n"})

    def summarize(self, raw, stderr=b""):
        from .trajectory import summarize_zcode
        return summarize_zcode(raw, stderr)
