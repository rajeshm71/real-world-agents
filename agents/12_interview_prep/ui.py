"""Gradio UI for the interview-prep agent.

Kept deliberately minimal: JD + background textboxes, a chat log, and
an answer box. If a reader wants to understand the adaptive multi-turn
technique, they should read `agent.py`'s `run_interview` -- this file
is UI glue, not the load-bearing code.

`run_interview` in agent.py is a single blocking call that drives an
entire session (it calls `input()` for each answer by default). A
Gradio callback fires once per submitted message and must return
immediately, so this file can't just call `run_interview` directly --
it re-implements the same one-turn-per-call loop body inline, reusing
every shared building block from agent.py (`_build_agent`,
`_status_line`, `_translate_api_error`, `_build_session_report`) so
the two-tier safety-cap logic isn't invented twice, just re-triggered
per Gradio event instead of per Python `while` iteration.

Launched via `uv run python -m agent --ui` or, when deployed to a
HuggingFace Space, as the container's entry point.
"""

from __future__ import annotations

import html
from pathlib import Path

import gradio as gr

# Dual-mode import (same rationale as agents #01/#02/#03/#04).
try:
    from .agent import (
        _DEFAULT_MODEL_BY_PROVIDER,
        DEFAULT_MAX_FOLLOWUPS_PER_TOPIC,
        DEFAULT_MAX_TOPICS,
        _build_agent,
        _build_session_report,
        _status_line,
        _translate_api_error,
        resolve_provider,
    )
    from .schemas import InterviewTurn, TurnRecord
except ImportError:
    from agent import (
        _DEFAULT_MODEL_BY_PROVIDER,
        DEFAULT_MAX_FOLLOWUPS_PER_TOPIC,
        DEFAULT_MAX_TOPICS,
        _build_agent,
        _build_session_report,
        _status_line,
        _translate_api_error,
        resolve_provider,
    )
    from schemas import InterviewTurn, TurnRecord

from common.llm import resolve_model

_EXAMPLES_DIR = Path(__file__).parent / "examples"


def _feedback_html(turn: InterviewTurn) -> str:
    fb = turn.feedback_on_previous_answer
    if fb is None:
        return ""
    strengths = "".join(f"<li>{html.escape(s)}</li>" for s in fb.strengths)
    gaps = "".join(f"<li>{html.escape(g)}</li>" for g in fb.gaps)
    return (
        f"<div style='padding:0.5em; margin:0.4em 0; background:#f0f0f0; "
        f"border-radius:6px; color:#111;'>"
        f"<strong>Score: {fb.score}/100</strong>"
        + (f"<ul>{strengths}</ul>" if strengths else "")
        + (f"<div><em>Gaps:</em><ul>{gaps}</ul></div>" if gaps else "")
        + f"<div><em>Tip:</em> {html.escape(fb.improvement_tip)}</div>"
        + "</div>"
    )


def _start_session(jd_text: str, background_text: str, max_topics: int, max_followups: int):
    """Build the agent, run turn 1, return the initial chat history +
    session state dict."""
    if not jd_text or not jd_text.strip():
        return [], None, "Paste a job description first."

    # Mirror the CLI's run_interview(), which rejects max_topics < 1 /
    # max_followups_per_topic < 0 with a clear error -- without this
    # check, typing 0 into the "Max topics" Gradio Number would silently
    # produce a one-topic session instead of an explicit rejection.
    # int(...) also guards against gr.Number's float return type.
    max_topics = int(max_topics)
    max_followups = int(max_followups)
    if max_topics < 1:
        return [], None, f"Max topics must be >= 1, got {max_topics}."
    if max_followups < 0:
        return [], None, f"Max follow-ups/topic must be >= 0, got {max_followups}."

    provider = resolve_provider()
    if provider == "mock":
        # Mock mode has no live turn-by-turn agent to drive in the UI --
        # the CLI's `--provider mock` path already covers the
        # deterministic-fixture use case (tests/CI). Tell the user
        # plainly rather than faking a chat session.
        return [], None, "LLM_PROVIDER=mock has no interactive session -- run the CLI to see the canned fixture, or set a real provider to use this UI."

    try:
        from agents import Runner
    except ImportError:
        return [], None, "openai-agents is not installed. Run `uv sync` at the workspace root."

    model = _DEFAULT_MODEL_BY_PROVIDER.get(provider) or resolve_model(provider)
    agent = _build_agent(model=model, provider=provider)

    jd_block = f"Job description:\n\n{jd_text.strip()}"
    if background_text and background_text.strip():
        jd_block += f"\n\nCandidate background:\n\n{background_text.strip()}"
    status = _status_line(
        topic_index=0, followups_this_topic=0,
        max_topics=max_topics, max_followups_per_topic=max_followups,
    )

    try:
        result = Runner.run_sync(agent, input=f"{jd_block}\n\n{status}")
        turn = result.final_output_as(InterviewTurn, raise_if_incorrect_type=True)
    except Exception as exc:
        err = _translate_api_error(exc)
        return [], None, err.message

    # The schema allows action="conclude" on ANY turn, including turn 1 --
    # only the system prompt says not to. agent.py's run_interview defends
    # against this with its topic_index==-1 forcing logic; this function
    # has no equivalent (it hardcodes topic_index=0 for turn 1 rather than
    # using that sentinel), so without this check a turn-1 "conclude"
    # would slip through with question=None (enforced by InterviewTurn's
    # cross-field validator) and display the literal string "None" as the
    # first question.
    if turn.question is None:
        return [], None, (
            "The model tried to end the interview before asking a single "
            "question. Please try again."
        )

    state = {
        "agent": agent,
        "topic_index": 0,
        "followups_this_topic": 0,
        "current_input": result.to_input_list(),
        "transcript": [TurnRecord(topic_index=0, turn=turn, candidate_answer=None)],
        "max_topics": max_topics,
        "max_followups_per_topic": max_followups,
        "done": False,
    }
    history = [{"role": "assistant", "content": f"**{turn.topic}**\n\n{turn.question}"}]
    return history, state, ""


def _submit_answer(answer: str, history: list, state: dict | None):
    """One Gradio submit == one turn of the interview. Re-implements
    run_interview's loop body (see module docstring) instead of
    calling it directly, since that function blocks on stdin."""
    if state is None or state.get("done"):
        return history, state, "", ""
    if not answer or not answer.strip():
        return history, state, "", "Type an answer first."

    history = [*history, {"role": "user", "content": answer}]

    max_topics = state["max_topics"]
    max_followups_per_topic = state["max_followups_per_topic"]
    # include_feedback_reminder=True: mirrors agent.py's run_interview
    # fix -- a real candidate answer was just submitted, so the model
    # needs the per-turn reminder to actually populate
    # feedback_on_previous_answer (see _status_line's docstring).
    status = _status_line(
        topic_index=state["topic_index"],
        followups_this_topic=state["followups_this_topic"],
        max_topics=max_topics,
        max_followups_per_topic=max_followups_per_topic,
        include_feedback_reminder=True,
    )
    next_input = [*state["current_input"], {"role": "user", "content": f"{answer}\n\n{status}"}]

    try:
        from agents import Runner
        result = Runner.run_sync(state["agent"], input=next_input)
        turn = result.final_output_as(InterviewTurn, raise_if_incorrect_type=True)
    except Exception as exc:
        err = _translate_api_error(exc)
        return history, state, "", err.message

    # Same two-tier cap backstop as run_interview's loop body.
    effective_action = turn.action
    topic_index = state["topic_index"]
    followups_this_topic = state["followups_this_topic"]
    if effective_action == "ask_followup" and followups_this_topic >= max_followups_per_topic:
        effective_action = "ask_new_topic"
    if effective_action == "ask_new_topic" and topic_index >= max_topics - 1:
        effective_action = "conclude"

    if effective_action == "ask_new_topic":
        topic_index += 1
        followups_this_topic = 0
    elif effective_action == "ask_followup":
        followups_this_topic += 1

    feedback_html = _feedback_html(turn)
    # candidate_answer is the answer to THIS turn's own question, which
    # doesn't exist yet (the user hasn't submitted it). None is honest
    # here -- _build_session_report never reads this field for
    # aggregation anyway (see agent.py's shift-by-one scoring note).
    transcript = [*state["transcript"], TurnRecord(topic_index=topic_index, turn=turn, candidate_answer=None)]

    if effective_action == "conclude":
        report = _build_session_report(transcript, "concluded" if turn.action == "conclude" else "cap_reached")
        summary_lines = [
            f"**Session complete -- overall readiness: {report.overall_readiness_score}/100**",
            "",
        ]
        for t in report.topics:
            summary_lines.append(f"- [{t.best_score}/100] {t.topic_label}")
        if report.top_improvement_areas:
            summary_lines.append("\n**Top improvement areas:**")
            summary_lines.extend(f"- {area}" for area in report.top_improvement_areas)
        history = [*history, {"role": "assistant", "content": feedback_html + "\n\n" + "\n".join(summary_lines)}]
        state["done"] = True
        return history, state, "", ""

    history = [*history, {"role": "assistant", "content": feedback_html + f"\n\n**{turn.topic}**\n\n{turn.question}"}]
    state.update(
        topic_index=topic_index,
        followups_this_topic=followups_this_topic,
        current_input=result.to_input_list(),
        transcript=transcript,
    )
    return history, state, "", ""


def build_ui() -> gr.Blocks:
    """Build and return the Gradio Blocks app. Kept as a function (not
    module-level) so importing this module doesn't spin up any UI
    state -- important for tests."""
    default_provider = resolve_provider()

    with gr.Blocks(title="Interview Prep", theme=gr.themes.Soft()) as app:
        gr.Markdown(
            "# Mock interview practice\n"
            "Paste a job description (and optionally your background); "
            "get an adaptive mock interview. The agent scores each answer, "
            "drills deeper when an answer is thin, and moves to a new "
            "requirement once you've demonstrated it.\n\n"
            f"**Provider:** `{default_provider}`. "
            "[Source code](https://github.com/rajeshm71/real-world-agents/tree/main/agents/12_interview_prep)"
        )

        state = gr.State(value=None)

        with gr.Row():
            with gr.Column(scale=1):
                jd_box = gr.Textbox(label="Job description", lines=12)
                background_box = gr.Textbox(label="Candidate background (optional)", lines=6)
                with gr.Row():
                    max_topics_box = gr.Number(label="Max topics", value=DEFAULT_MAX_TOPICS, precision=0)
                    max_followups_box = gr.Number(
                        label="Max follow-ups/topic", value=DEFAULT_MAX_FOLLOWUPS_PER_TOPIC, precision=0
                    )
                start_btn = gr.Button("Start interview", variant="primary")
                error_box = gr.Markdown()
            with gr.Column(scale=2):
                chatbot = gr.Chatbot(label="Interview", type="messages", height=500)
                answer_box = gr.Textbox(label="Your answer", lines=3)
                submit_btn = gr.Button("Submit answer")

        start_btn.click(
            _start_session,
            inputs=[jd_box, background_box, max_topics_box, max_followups_box],
            outputs=[chatbot, state, error_box],
        )
        submit_btn.click(
            _submit_answer,
            inputs=[answer_box, chatbot, state],
            outputs=[chatbot, state, answer_box, error_box],
        )
        answer_box.submit(
            _submit_answer,
            inputs=[answer_box, chatbot, state],
            outputs=[chatbot, state, answer_box, error_box],
        )

    return app


if __name__ == "__main__":
    build_ui().launch()
