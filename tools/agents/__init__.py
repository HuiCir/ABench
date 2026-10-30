from .claude import Claude
from .codex import Codex
from .dsh import Dsh
from .gemini import Gemini
from .offline import Command, Mock, Oracle
from .zcode import ZCode

REGISTRY = {a.name: a for a in (Claude(), Codex(), Gemini(), Dsh(), ZCode(), Oracle(), Mock(), Command())}
# Harnesses with a model and a pinned CLI in their own agent image
HARNESSES = tuple(name for name, agent in REGISTRY.items() if agent.protocols)

def get_agent(name):
    try:
        return REGISTRY[name]
    except KeyError:
        raise ValueError(f"unknown agent {name!r}; choose {', '.join(REGISTRY)}") from None
