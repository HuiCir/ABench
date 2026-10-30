import json
from .base import Agent, Launch, pinned


class Gemini(Agent):
    """Google Gemini CLI (`gemini --output-format stream-json`, headless because stdin is not a terminal).
    Usage statistics, telemetry and update checks are off; effort sets the model's thinking level."""
    name = "gemini"
    version = pinned("gemini")["version"]
    protocols = ("gemini",)
    default_model = "gemini-4-argon"
    needs_network = True
    parameters = ("effort",)
    # google_web_search is search grounding at Google; web_fetch reads public pages through the API's URL context
    disabled_tools = ("google_web_search", "web_fetch")
    efforts = ("minimal", "low", "medium", "high")
    session_logs = (".gemini/tmp",)

    def effort_values(self, endpoint):
        # Gemini 2.5 takes a thinking budget, not a level; minimal and medium exist only for Flash models.
        model = str(endpoint.get("model", "")).lower()
        if "2.5" in model:
            return ()
        return self.efforts if "flash" in model else ("low", "high")
    cli_version_argv = ("gemini", "--version")

    def credential_env(self, endpoint):
        return ("GEMINI_API_KEY",)

    def launch(self, args, paths, prompt, endpoint=None):
        options = self.options(endpoint)
        settings = {"security": {"auth": {"selectedType": "gemini-api-key"}},
                    "privacy": {"usageStatisticsEnabled": False}, "telemetry": {"enabled": False},
                    "general": {"disableAutoUpdate": True, "disableUpdateNag": True},
                    "tools": {"exclude": list(self.disabled_tools)}}
        if options.get("effort"):
            settings["modelConfigs"] = {"customOverrides": [{
                "match": {"model": endpoint["model"]},
                "modelConfig": {"generateContentConfig": {"thinkingConfig": {"thinkingLevel": str(options["effort"]).upper()}}}}]}
        argv = ["gemini", "--output-format", "stream-json", "--yolo", "--skip-trust", "-m", endpoint["model"],
                "--include-directories", str(paths["view"])]
        return Launch(argv, {"GOOGLE_GEMINI_BASE_URL": endpoint["base_url"]}, prompt.encode(),
                      {".gemini/settings.json": json.dumps(settings, indent=2) + "\n"})

    def summarize(self, raw, stderr=b""):
        from .trajectory import summarize_gemini
        return summarize_gemini(raw, stderr)
