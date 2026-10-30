import re
import shlex
from .base import Agent, Launch

class Command(Agent):
    name = "command"
    def launch(self, args, paths, prompt, endpoint=None):
        if not args.command:
            raise ValueError("command agent needs --command")
        values = {"ABENCH_" + k.upper(): str(v) for k, v in paths.items()}
        values.update(ABENCH_SUBMISSION_DIR=str(paths["submission"]), ABENCH_VIEW_DIR=str(paths["view"]),
                      ABENCH_DATA_DIR=str(paths["view"] / "data"), ABENCH_CHECKPOINT_DIR=str(paths["checkpoint"]))
        return Launch([re.sub(r"\$\{?(ABENCH_[A-Z_]+)\}?", lambda m: values.get(m[1], m[0]), t)
                       for t in shlex.split(args.command)], stdin=prompt.encode())

class Mock(Agent):
    name = "mock"
    def launch(self, args, paths, prompt, endpoint=None):
        return Launch(["bash", "/abench/control/mock_agent.sh"], stdin=prompt.encode())

class Oracle(Agent):
    name = "oracle"
    def launch(self, args, paths, prompt, endpoint=None):
        return Launch(["python3", "/abench/control/oracle.py"])
