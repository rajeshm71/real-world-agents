"""Root-level pytest conftest.

Purpose: pre-load the openai-agents SDK (pip-installed `agents` package)
into `sys.modules` BEFORE pytest walks the workspace `agents/` directory
as a namespace package. Without this pre-load, pytest's collection walk
populates `sys.modules['agents']` as the workspace's empty namespace
package -- and every subsequent `from agents import Agent` from within
an agent's test file fails with "cannot import name 'Agent' from
'agents' (unknown location)" because the namespace package doesn't have
those symbols.

Agents #04 and #08 use openai-agents; agents that don't use it get no
side effect from this file since the try/except silently absorbs the
ImportError.

This file also documents the reason the `tests/__init__.py` files in
each agent directory were REMOVED: with them present + pytest's
--import-mode=importlib (see pyproject.toml's addopts), every agent's
`test_smoke.py` collapsed to a shared `tests.test_smoke` module name
in sys.modules -- so only the first-collected file (#01's) actually
ran, and every subsequent agent's test file was silently aliased to
#01's tests. Removing tests/__init__.py lets pytest give each file
a unique path-based module identity.
"""

try:
    import agents  # noqa: F401 -- side effect: caches SDK in sys.modules
except ImportError:
    # openai-agents not installed. Fine; agents that don't need it
    # (#01, #02, #03, #05, #06, #07) still work. Agents that do need
    # it (#04, #08) will get their own clear ImportError when they
    # try to import it themselves.
    pass
