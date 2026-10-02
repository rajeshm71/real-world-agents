# Interview prep agent

Paste a job description (and optionally your own background); get an
adaptive mock interview. The agent asks a question grounded in a real
JD requirement, scores your answer honestly, and decides live whether
to drill deeper on the same topic or move to a fresh requirement --
until it's covered a good spread of the role or hits a safety cap.

## Technique demonstrated

**Adaptive multi-turn dialogue with real-time answer assessment**, via
the [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/).
Every other OpenAI-Agents-SDK agent in this catalog (#04, #08) runs
its entire loop inside ONE `Runner.run_sync` call, against a
deterministic tool (meeting notes, a pytest sandbox). This agent's
loop is different in kind: a real human has to answer between every
step, so the "loop" is our own Python `while`, one `Runner.run_sync`
call per turn, with conversation state threaded across those separate
calls via the SDK's `RunResult.to_input_list()`:

    Turn 1: agent asks Q1 (grounded in a JD requirement)
    -> candidate answers A1
    Turn 2: agent scores A1, picks ask_followup / ask_new_topic /
            conclude, asks Q2 if not concluding
    -> candidate answers A2
    Turn 3: agent scores A2, ...

A real run against the shipped example JD (backend engineer role)
looked like this -- thin first answer triggers a follow-up, a
stronger answer moves the interview on:

    TOPIC: API Design and Evolution
    Q: Can you describe your approach to designing and evolving REST
       APIs, particularly how you handle versioning and backward
       compatibility?
    A: I've worked with APIs before.

    TOPIC: API Design and Evolution                 [score=10 on above]
    Q: Could you provide a specific example... please explain the
       versioning strategy you implemented...
    A: At my last job I owned the merchant payouts API. We went from
       v1 to v2 with a deprecation window for old clients...

    TOPIC: Production Reliability and On-Call Experience  [score=70]
    Q: Can you walk me through another incident you handled...

Two code-owned counters (`topic_index`, `followups_this_topic`) bound
the session even if the model ignores its own budget -- see "Where
this fails" for the one honest rough edge this produces.

## Why this technique for this use case

A real interview doesn't run off a fixed question list: how deep to
drill and when to move on depends on how well the candidate actually
answered. That's inherently a human-in-the-loop technique, not a
single-shot extraction or a tool-use loop -- which is why this agent
carries no tools at all (the JD/background text is already in the
prompt context; there's nothing external to ground against) and no
`max_turns` ReAct cap (a human answers between every step, so the
SDK's own turn limit doesn't apply the way it does for #04/#08).

Where this technique is NOT the right fit: anything that should run
unattended (batch-scoring a pile of pre-written answers wants #11's
batch+ranking pattern, not a live back-and-forth), or anything needing
cross-provider portability today (Anthropic/Gemini via LiteLLM is a
documented one-line swap, not implemented in v1 -- same stance as
#04/#08).

Cost is low: each turn is one `gpt-4.1-mini-2025-04-14` call
(`$0.40`/1M input, `$1.60`/1M output -- see `common/pricing.py`), and
even a full 8-turn session (the real run above) stays well under a
cent -- conversation history grows turn over turn, but nowhere near
enough to matter at these rates.

## What it does

Input: a job-description text file, and an optional candidate-
background text file (a short bio or resume snippet). Output: a live,
turn-by-turn mock interview in the terminal, ending in a structured
report (topics covered, per-topic best score, an overall readiness
score, and the improvement areas from your weakest topics) written to
`last_run.json`.

## How to run locally

```bash
git clone https://github.com/rajeshm71/real-world-agents.git
cd real-world-agents
cp .env.example .env    # set OPENAI_API_KEY
cd agents/12_interview_prep
uv run python -m agent examples/sample_jd_backend_engineer.txt \
    --background examples/sample_background.txt
```

Answer each question as it's asked; the session ends when the agent
concludes (or a safety cap is hit -- see "Code walkthrough"). No API
key needed to explore the code or run tests: `LLM_PROVIDER=mock uv run
python -m agent examples/sample_jd_backend_engineer.txt` returns a
canned 4-turn session instantly. `--ui` launches a Gradio chat version
of the same thing (real providers only; mock mode has no live session
to drive -- see `ui.py`'s `_start_session`).

Flags: `--background <path>`, `--max-topics` (default 5),
`--max-followups` (default 2), `--model`, `--provider`
(`openai`/`ollama`/`mock`).

## Code walkthrough

1. `agent.py::run_interview()` -- the whole technique in one function:
   resolves the provider, validates the JD, builds the Agent, then
   drives the turn-by-turn `while` loop. Read this first.
2. `agent.py`'s two-tier safety-cap block inside that loop (search for
   "effective_action") -- the deterministic backstop half of the cap
   mechanism; `_status_line()` above it is the proactive half (told to
   the model every turn so the backstop rarely has to fire).
3. `agent.py::_build_session_report()` -- pure-Python aggregation over
   the recorded turns, no extra LLM call. The scoring here is a
   shift-by-one pairing (`transcript[i+1]`'s feedback scores
   `transcript[i]`'s answer) -- read the docstring before touching
   this function, it's the one non-obvious piece of arithmetic in the
   whole agent.
4. `agent.py::_mock_result()` -- a hand-built 4-turn canned transcript
   fed through the SAME `_build_session_report()` the real path uses,
   so mock mode exercises the real aggregation logic instead of
   duplicating it.
5. `agent.py::_translate_api_error()` -- R5 case 3 (API failure), same
   6-branch shape as #04/#08: class-name first, status-code second,
   message-fallback, Ollama-connection-refused hint, generic fallback.
6. `agent.py::_looks_like_a_jd()` + the `max_topics`/
   `max_followups_per_topic` validation right after it -- R5 case 1
   (bad input), checked before any Agent construction.
7. `schemas.py::InterviewTurn` -- the SDK's `output_type`. Note
   `topic` is documented as display-only; `TurnRecord.topic_index` (in
   the same file) is the loop-owned counter that's actually load-
   bearing for the cap logic.
8. `ui.py` -- re-implements one turn of `run_interview`'s loop body
   per Gradio submit event (a Gradio callback can't block on
   `input()` the way the CLI's default `answer_source` does) --
   read its module docstring for why this couldn't just call
   `run_interview` directly.

## When to use / When NOT to use

**Use when:**
- You're prepping for a specific role and want practice questions
  grounded in that JD's actual requirements, not generic interview
  question banks.
- You want honest, specific feedback per answer (what was strong, what
  was missing) rather than a pass/fail verdict.
- You're building a similar "the depth of the interaction should adapt
  to the user's own input, turn by turn" tool and want a working
  reference for the SDK's multi-turn continuation mechanism.

**When NOT to use:**
- Scoring many candidates' pre-written answers in bulk -- that's
  batch+ranking (#11), not a live dialogue.
- You need the interview to follow a fixed, auditable question script
  (e.g. structured/legally-defensible hiring interviews) -- this agent
  adapts its questions live, which is the opposite of a fixed script.
- You need multi-language support -- the prompt and heuristics are
  English-only in v1.

## Where this fails

- **Vague JDs produce vague questions.** A JD heavy on "team player" /
  "fast-paced environment" boilerplate gives the model little to
  ground `requirement_from_jd` in -- questions degrade toward generic
  behavioral filler. Symptom: `requirement_from_jd` reads like a
  paraphrase of company-culture copy rather than a concrete skill.
- **Topic labels can be inconsistent even within one logical topic.**
  Observed in a real run: the model treated a follow-up question as
  `action="ask_followup"` (correctly NOT starting a new topic-cap
  slot) but gave it a completely different display label than the
  topic's earlier turns ("Production Reliability..." then "Postgres at
  Scale..." for what was structurally the same `topic_index`). The
  cap bookkeeping stayed exactly correct throughout (verified: 4
  distinct `topic_index` groups for `max_topics=4`, not 5) -- but the
  final report's per-topic label is just whichever label the model
  used LAST for that group, which can look like a mismatch to a human
  skimming the summary. This is the direct, working-as-designed
  consequence of `topic` being display-only (see schemas.py) --
  documented rather than hidden.
- **The nudge-then-force safety valve can silently override the
  model's stated action.** If the model ignores the live budget told
  to it every turn (see `_status_line`) and tries to exceed a cap
  anyway, the loop forces the bookkeeping (topic change, or conclude)
  while still showing the model's own question/feedback text as-is.
  The model's own `turn.action` field (preserved in the transcript)
  and what actually happened can diverge in this edge case. Rare in
  practice -- across every manually verified run, the model stayed
  within budget on its own -- but real, and worth knowing about if
  you're reading `transcript` programmatically rather than through
  `_build_session_report`.
- **No dedicated contradiction-detection across answers.** If the
  candidate contradicts an earlier answer, the SDK's own conversation
  history is all the model has -- there's no #16-style verification
  step checking answers against each other.
- **`max_topics=5` is a coarse proxy for interview length.** A real
  45-minute technical interview trades depth against breadth
  differently than a fixed topic-count cap does.
- **Local models needed a per-turn instruction reminder to score
  answers at all -- fixed, not just documented.** Manually verified
  live against three local models (`gemma4:e4b` -- the catalog
  default, `qwen2.5vl:7b`, `qwen2.5:7b`). Branching was reliable from
  the start: across a real multi-turn session, `gemma4:e4b` correctly
  drilled into a thin answer ("I worked on some APIs."), moved on
  after strong answers, and caught an off-topic answer and circled
  back to it. But `feedback_on_previous_answer` initially came back
  `null` on every non-first turn, across all three models, even
  though the system prompt requires populating it on "every later
  turn." Root-caused with a direct experiment rather than assumed:
  a single isolated prompt (no conversation history) got the model to
  populate the field correctly every time, proving it wasn't a
  capability gap or a schema issue (`AgentOutputSchema`'s generated
  JSON schema correctly lists the field as `required` with
  `anyOf: [AnswerFeedback, null]`). The actual cause: this model's
  adherence to the system prompt (set once, at Agent construction)
  measurably degrades once even one turn of conversation history has
  accumulated -- an instruction stated only in the system prompt does
  not reliably survive to turn 2 onward. The fix was to restate the
  instruction directly in the per-turn user message via
  `_status_line`'s `include_feedback_reminder` flag, the same
  mechanism that already reinforces the cap rules every turn -- once
  applied, `gemma4:e4b` populates real scores and specific gaps on
  every turn, verified end-to-end through the actual `run_interview()`
  code path (not a standalone script). `qwen3.5:9b` (a reasoning
  model) still produced no visible output at all within a normal
  token budget over this agent's OpenAI-compat surface -- consistent
  with this catalog's known finding that qwen3.x reasoning models
  exhaust their budget in hidden `<think>` blocks unless routed
  through `common/ollama_nothink_proxy.py`, which wasn't tried here.
  A second, related gap surfaced once feedback was actually populating:
  scores clustered tightly in a 2-4 (out of 100) band regardless of
  answer quality -- the qualitative text for one turn literally said
  "this was an excellent answer that directly addressed the prompt's
  core technical requirement" while the numeric score was 4. Initially
  assumed to be a model-calibration limitation and documented as such;
  turned out to be the SAME root cause as the null-feedback gap above,
  just for a different instruction. An isolated single-call test
  scored a genuinely excellent answer 90/100 -- proving the model can
  use the full 0-100 range -- while the same answer through the real
  multi-turn flow scored 4/100, confirming the scale instruction (like
  the "must not be null" instruction) wasn't surviving past turn 1
  either. A/B tested directly: identical conversation history,
  identical answer, only the reminder text changed -- adding explicit
  0-100 anchor points (0-20 no real answer, 21-40 vague, 41-60
  adequate, 61-80 solid, 81-100 exceptional) to the same per-turn
  reminder took the score from 4 to 82. Fixed the same way: the score-
  scale anchors are now part of `_status_line`'s feedback reminder
  (and mirrored in the schema's field description and the system
  prompt for consistency). Re-verified end-to-end through the real
  `run_interview()` code path: the same scripted session that
  previously scored every topic 3-4/100 now scores 85/100, 78/100, and
  15/100 across strong, solid, and off-topic answers respectively --
  correctly differentiated, with an `overall_readiness_score` of
  59/100 that actually reflects the mix of answers given.
