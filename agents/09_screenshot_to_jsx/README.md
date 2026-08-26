# Screenshot -> React JSX (multimodal reconstruction)

Upload a full-page screenshot of a UI. Get back a React functional component that reconstructs the visible structure. Ships JSX code, detected sections, honest notes on what the model had to guess, and a list of npm packages you'll need to install. A deployable OSS demo for anyone who wants to see the vision-input + structured-output pattern applied to a code-generation domain, plus a starting point for "convert this design mockup into React scaffolding I can iterate on."

## Verification status

| Path | Status |
|---|---|
| Mock mode (`LLM_PROVIDER=mock`) | Fully covered by 35-test suite |
| Schema validators (component_name PascalCase, jsx_code cross-check, imports) | Fully covered, both directions |
| Real vision generation (`LLM_PROVIDER=anthropic\|openai\|gemini` + key) | **Not yet verified against a live API call.** Structural correctness proven via #01's identical Instructor + vision pattern (which IS shipping in production), but no end-to-end run has been billed yet. Same open-item status as every other agent's first ship. |
| Render + compare + iterate (Phase B/C) | **Queued.** v0.1 ships Phase A (one-shot generation only). See "Roadmap" section below. |

## Technique demonstrated

**Multimodal structured extraction applied to a code domain.** The pipeline is exactly agent #01's (Instructor + vision + Pydantic response_model + multi-provider `from_provider()`), but the load-bearing schema field is a code string that has to be syntactically valid JSX AND coherent with the declared `component_name`. The schema validators enforce that at parse time so a `component_name="AdminDashboard"` result whose JSX defines `Homepage` gets rejected rather than silently shipping a broken import.

Distinct from every other agent: #01 extracts numbers + strings, #02/#03/#04/#06 extract structured records, #05 collaborates on prose, #07 optimizes a prompt, #08 generates test code with subprocess feedback. **#09 is the first agent whose output is a code string that will be dropped into a foreign project (a React codebase) and expected to build.**

## Why this technique for this use case

Full-page screenshot reconstruction is a decisions-heavy task: what to name the component, what layout technique to use, which elements to identify as sections, how to handle assets the model can't see clearly, which npm packages the caller will need. Free-form text output ("here's a React component:") gives you inconsistent shape a downstream tool can't reliably parse. A Pydantic schema locks the output structure so the Gradio UI, a batch script, and a code-review pipeline all get the same shape regardless of which provider is behind it.

Multi-provider matters here because vision-model quality for structured markup varies more than for extraction: Anthropic's Claude tends to produce cleaner semantic HTML than OpenAI or Gemini, but every provider works via the same `from_provider()` call. Anthropic is the default; every provider is a one-env-var switch.

Where this technique is NOT the right fit: (a) pixel-perfect reconstruction (any vision model will miss subtle spacing, exact colors, font choices); (b) real asset extraction (screenshots don't carry source URLs; the agent uses div placeholders documented in `notes`); (c) sensitive designs (screenshots leave your machine and go to whichever provider you selected).

## What it does

Input: raw bytes of a screenshot (PNG/JPEG/WebP/GIF). Output: a validated `ReconstructedComponent` Pydantic object with:

- `component_name` (PascalCase, JS-identifier-valid)
- `jsx_code` (a functional-component code string, cross-checked to actually define `component_name`)
- `imports` (npm packages beyond `react` you'll need to install)
- `styling_approach` (`"tailwind"` or `"inline_styles"` per your choice)
- `notes` (honest notes on what the model had to assume, guess, or skip)
- `detected_sections` (top-level regions the model identified)

Under `LLM_PROVIDER=mock` the whole thing returns a canned `MockLandingPage` component (header + main + footer, valid JSX, byte count echoed into `notes`) with no network call.

## How to run locally

Four commands from a fresh clone (`python -m agent` must run from inside the agent's own directory: `agent` is a submodule of the digit-prefixed `09_screenshot_to_jsx` package):

```bash
git clone https://github.com/rajeshm71/real-world-agents.git
cd real-world-agents
cp .env.example .env    # set ANTHROPIC_API_KEY (default) or LLM_PROVIDER=mock
cd agents/09_screenshot_to_jsx
```

Mock demo (no API key, canned result, works with the shipped example screenshots):

```bash
LLM_PROVIDER=mock uv run python -m agent examples/landing_page.png
```

Real reconstruction against the shipped landing-page mock:

```bash
uv run python -m agent examples/landing_page.png
```

Rough cost at Anthropic Claude Sonnet-5 pricing: **~$0.01-0.03 per generation** for a full-page screenshot. Cheaper on `gpt-4.1-mini` (~$0.005) at some quality cost.

Switch providers with `LLM_PROVIDER=openai` or `LLM_PROVIDER=gemini`:

```bash
uv run python -m agent examples/dashboard.png --provider openai --styling inline_styles
```

Gradio UI (file upload + sample-picker):

```bash
uv run python -m agent --ui
```

Every real CLI run writes the full `ReconstructedComponent` JSON to `last_run.json` next to `agent.py` (gitignored via `agents/*/last_run.json`).

## Code walkthrough

Under 500 LOC excluding UI. Read these in order:

1. **`schemas.py`**: `ReconstructedComponent` Pydantic model. `component_name` field validator enforces PascalCase (regex `^[A-Z][A-Za-z0-9]*$`); `imports` field validator rejects empty-string entries that would produce broken `import ''` statements. `@model_validator` cross-checks that `jsx_code` actually defines a function or const matching `component_name` -- so a `component_name="AdminDashboard"` output whose JSX defines `Homepage` gets rejected at parse time. Guards against "confidently wrong" hallucinations that would silently ship a broken import to the caller.
2. **`prompts/system.txt`**: instructs the model to scan top-to-bottom, use semantic HTML, use `<div>` placeholders for assets it can't see clearly, and be honest about assumptions in `notes`. Explicit "do not return a stub" rule to prevent the model from just producing `function X() { return <div>TODO</div>; }` on hard inputs.
3. **`agent.py::resolve_provider()`**: reads `LLM_PROVIDER` (default `"anthropic"`, not `"openai"` -- see module docstring for why Claude wins on markup output).
4. **`agent.py::reconstruct()`**: the public API. Fail-fast validation before any FS read or LLM call (zero-byte input, unknown media_type, unknown styling), then either short-circuits to `_mock_reconstruction()` or routes through `_get_real_client()` + `_build_content()` + `client.create(response_model=ReconstructedComponent, ...)`.
5. **`agent.py::_build_content()`**: provider-branching image + text content-block builder. Anthropic uses `type: "image"` with `source.type: "base64"`; OpenAI + Gemini use `image_url` with a `data:` URI. Same structure as agent #01's `_build_content` -- both agents share the pattern because both feed image bytes to a vision model.
6. **`agent.py::_get_real_client()`**: one-line Instructor client via `instructor.from_provider(f"{prefix}/{model}")`. Lazy import so mock-mode CI doesn't need Instructor or any provider SDK installed.
7. **`agent.py::_translate_api_error()`**: R5 case 3. Seven-branch priority order: ValidationError first (attaches `partial` with `raw_output` so a caller can inspect what the model produced after all retries exhausted), then the standard 6-branch class-name -> status -> message -> generic priority same as agents #02-#08.
8. **`agent.py::_mock_reconstruction()`**: deterministic canned `MockLandingPage` component; input byte count encoded into `notes` so tests can prove the mock saw its input. Two variants (tailwind + inline_styles) so the styling override is testable end-to-end without a real API call.
9. **`ui.py::build_ui()`**: Gradio Blocks. Image upload + styling Radio on the left, Run button. Right column: component name, detected sections chips, npm imports list, syntax-highlighted JSX code block, model's notes below. Sample-picker widget hydrates from `examples/`.
10. **`tests/test_smoke.py`**: 35 tests, all under `LLM_PROVIDER=mock`. Covers mock round-trip + byte-echo + JSON serializability, all schema validator branches (both directions of the component_name cross-check), all three provider content-shape variants (Anthropic image block vs. OpenAI image_url), all R5 branches including the 6-branch translator + validation-error partial attachment, provider resolution, media-type guessing, and examples/README sanity.

## When to use / When NOT to use

**Use when:**
- You have a full-page UI screenshot (design mockup, competitor page, existing product) and want React JSX scaffolding as a starting point you'll iterate on
- You want to see the vision + structured-output pattern applied to code generation
- You're comfortable spending ~$0.01-0.03 per generation for Anthropic-quality output
- You'll manually review + adjust the output before using it (this is a first draft, not a shipping tool)

**Do NOT use when:**
- You need pixel-perfect reconstruction (fundamentally beyond any vision model today)
- You need real asset extraction (logos, images, backgrounds): screenshots don't carry source URLs
- Your designs contain sensitive/confidential content: the screenshot goes to the provider you selected
- You need production-quality code: the output is a first draft that assumes structure, uses generic placeholders, and makes styling choices you may not want

## Where this fails

**Specific, honest failure modes:**

- **Assets the model can't see clearly**: logos, icons, background images, user avatars. The prompt tells the model to use `<div>` placeholders sized like the original and flag them in `notes`. A model following instructions produces coherent scaffolding with `{/* logo image goes here */}` comments; a model getting creative might invent a `<img src="/assets/company-logo.png" />` reference to a file that doesn't exist. Read `notes` before trusting the output.
- **Complex CSS Grid layouts**: the model tends to use `flex` even when the original is a grid, especially for irregular grid patterns (masonry, spanning cells). Symptom: reconstruction looks right for the first few rows, then diverges as the model tries to force a grid into flex boxes. Workaround: none in v0.1; Phase C's iteration loop will help here.
- **Media queries and responsive breakpoints**: the model sees ONE screenshot at ONE viewport. It has no way to infer mobile/tablet breakpoints from a single desktop screenshot. Any responsive design in the output is guessed from screenshot proportions and Tailwind's default breakpoints, not observed from the input.
- **Component decomposition**: the output is ONE monolithic component per screenshot. A real React codebase would extract `<Card>`, `<Button>`, `<NavItem>` etc. as reusable sub-components. Post-processing (either by the reader or a v1.1 refactor pass) is required for anything shipping-quality.
- **Styling approach mismatch with your project**: the agent picks Tailwind or inline styles per the `styling` flag, but your project might use CSS Modules, styled-components, emotion, or Vanilla Extract. Output has to be adapted; documented in `styling_approach` so a batch pipeline can filter.
- **Model returns generic component names**: if the screenshot doesn't obviously suggest a name (a partial UI with no header), the model may return `MyComponent` or `App`. The prompt explicitly says otherwise but doesn't fully prevent it. Workaround: rename in your editor before dropping in.
- **Validation-retry exhaustion**: on rare inputs (very sparse screenshots, ambiguous content) the model may produce output that fails schema validation across all `max_retries` attempts. `ScreenshotToJsxError` is raised with `partial.raw_text` set so you can see what the model was trying to say. Increase `--max-retries` or simplify the input.
- **Rate limit / API failure**: no auto-retry with backoff (explicit design decision). One failed vision call gets translated to "temporarily rate-limited" and the user re-runs. Silent retries would multiply the bill on the expensive part.
- **Real vision generation not yet verified**: this agent's code paths are structurally correct against Instructor + vision (same shape shipping in agent #01), but no live API call has been exercised at ship time. Same open-item status as every prior agent's first ship. Set `LLM_PROVIDER=anthropic` + `ANTHROPIC_API_KEY` and try it on a real screenshot to find any provider-specific quirks.

## Roadmap (phased delivery)

- **v0.1 (this release, Phase A):** multi-provider one-shot generation. What ships today.
- **v0.2 (Phase B, future release):** Playwright + Babel Standalone HTML wrapper pipeline that renders the produced JSX back to a fresh screenshot. Adds a `render_component` step + side-by-side "input vs. rendered" in the UI. `playwright install chromium` becomes a documented post-install step.
- **v1.0 (Phase C, future release):** LLM-as-judge comparison + iteration loop. Sends the original + rendered screenshots back to the model, gets structured critique, feeds it into the next iteration's prompt. Caps at `max_iterations` (default 3, ~$0.05-0.30 per full-iteration cycle).

Each phase is independently useful. Phase A gives you a JSX draft to iterate on manually. Phase B lets you eyeball the reconstruction without leaving the UI. Phase C automates the refinement.
