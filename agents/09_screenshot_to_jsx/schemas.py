"""Pydantic schema for the screenshot -> React JSX reconstructor.

`ReconstructedComponent` is what the vision model produces given a
full-page screenshot. It carries the JSX code itself plus structured
metadata a caller needs to use it: the component name, the styling
approach the model chose, the imports it assumed, sections it
identified, and a `notes` field where the model is instructed to be
honest about assumptions (missing assets, ambiguous elements, etc.).

Two model validators guard against the two most common "confidently
wrong" outputs:
1. `component_name` must be a valid PascalCase JS identifier.
2. `jsx_code` must actually define a function or const with that
   name; otherwise the caller's `import { <name> } from '...'` won't
   resolve.

Note: `styling_approach` is a two-value `Literal["tailwind",
"inline_styles"]`. `css_modules` was considered and dropped: the prompt
only instructs the model to produce Tailwind or inline styles, so
adding `css_modules` to the Literal would allow an unreachable state
the model would never legitimately hit. If a v1.1 adds CSS Modules
generation, extend both the Literal and the prompt together.
"""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

_PASCAL_CASE_RE = re.compile(r"^[A-Z][A-Za-z0-9]*$")


class ReconstructedComponent(BaseModel):
    """One vision-model pass's result. The JSX code is meant to be
    dropped into a caller's React project as a component file."""

    component_name: str = Field(
        ...,
        min_length=1,
        description=(
            "PascalCase component name, e.g. 'LandingPage'. Must be a "
            "valid JavaScript identifier so a caller can import it "
            "as `import LandingPage from './LandingPage'`."
        ),
    )
    jsx_code: str = Field(
        ...,
        min_length=1,
        description=(
            "The full JSX code as a string. Contains either "
            "`function <component_name>()` or `const <component_name> = `"
            "(the model may pick either form). Cross-checked by the "
            "validator below against component_name."
        ),
    )
    imports: list[str] = Field(
        default_factory=list,
        description=(
            "npm package names the caller will need to install. Empty "
            "for a pure-React component. `react` is implicit and does "
            "not need to appear."
        ),
    )
    styling_approach: Literal["tailwind", "inline_styles"] = Field(
        ...,
        description=(
            "Which styling technique the JSX uses. Callers who want to "
            "swap approaches (e.g. Tailwind -> CSS Modules) can filter "
            "on this field."
        ),
    )
    notes: str = Field(
        default="",
        description=(
            "Model's honest notes on what it had to assume, guess, or "
            "leave out (missing logo assets, blurred text, ambiguous "
            "component boundaries, etc.). Read this before trusting the "
            "output."
        ),
    )
    detected_sections: list[str] = Field(
        default_factory=list,
        description=(
            "Top-level regions the model identified, e.g. "
            "['header', 'hero', 'features', 'footer']. Helps a caller "
            "eyeball whether the model saw the same structure they did."
        ),
    )

    @field_validator("component_name")
    @classmethod
    def _component_name_is_pascal_case(cls, v: str) -> str:
        if not _PASCAL_CASE_RE.match(v):
            raise ValueError(
                f"component_name {v!r} is not a valid PascalCase JS "
                "identifier. Must start with an uppercase letter and "
                "contain only alphanumerics."
            )
        return v

    @field_validator("imports")
    @classmethod
    def _imports_are_non_empty(cls, v: list[str]) -> list[str]:
        """Empty strings in the imports list would produce broken
        `import '' from ''` statements in a caller's code. Reject."""
        for imp in v:
            if not imp or not imp.strip():
                raise ValueError(
                    "imports list contains an empty / whitespace-only "
                    "entry; every import must be a real npm package name"
                )
        return v

    @model_validator(mode="after")
    def _jsx_code_defines_component_name(self) -> ReconstructedComponent:
        """Cross-field check: the JSX code must actually declare a
        function or const with the claimed component_name. Guards
        against the "confidently wrong" case where the model returns
        `component_name='LandingPage'` but the JSX defines `Homepage`
        -- the caller's import would fail silently."""
        function_re = re.compile(
            rf"\bfunction\s+{re.escape(self.component_name)}\b"
        )
        const_re = re.compile(
            rf"\bconst\s+{re.escape(self.component_name)}\s*="
        )
        if not (function_re.search(self.jsx_code) or const_re.search(self.jsx_code)):
            raise ValueError(
                f"jsx_code does not define a function or const named "
                f"{self.component_name!r}. Model returned mismatched "
                "component_name and jsx_code; caller's import would fail."
            )
        return self
