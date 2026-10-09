# Sales outreach agent → qualifies leads, drafts and sends real email, handles replies

Give it a list of leads, a product description, and an ideal-customer-profile. It decides who's worth contacting, writes a personalized email for each one, and actually sends it through your own email account -- then, whenever a reply comes in, it classifies what the reply means and decides what happens next.

## Technique demonstrated

**A LangGraph node with a genuine external side effect** -- a real SMTP send -- inside a qualify → conditional-skip/draft → send pipeline, paired with a small cross-process JSON file that lets a second phase run independently, whenever a reply actually arrives (days later, a separate command, not a continuation of the same run).

Agent #03 is the only other LangGraph agent with a conditional branch (its retry loop), and agent #17 is the only one with a multi-way fan-out. Neither has a node that reaches outside the process and does something real. This one does: `send_email_node`'s job is to call the real SMTP client, not another LLM. And no other agent in this catalog runs in two separate sessions tied together by a state file -- every other agent's `last_run.json` is write-once and never read back.

## Why this technique for this use case

Real outbound email has two genuinely separate moments: send now, and react to a reply whenever it shows up, which could be minutes or weeks later. No single run of any agent can model that gap -- there has to be something written down in between. A small JSON file keyed by lead does the job without needing a database or a server. And the qualify step matters because not every lead is worth a personalized email: skipping the ones that don't fit means the model only has to write (and you only have to pay for) emails that are actually worth sending.

Where this technique is NOT the right fit: if you need the system to watch your inbox and react to replies on its own -- this agent only reacts when you paste a reply in yourself. Real inbox polling needs IMAP/OAuth access to a live mailbox, which is a different, bigger piece of infrastructure than this agent takes on.

## What it does

Input: a list of leads (company, contact, a few real specifics about them), who the email is from, what you're selling, and who your ideal customer is.

**Phase 1** (`send`): for each lead, decide if they fit well enough to contact. If yes, write a short personalized email that references something specific about them, then send it -- or just show you the draft, if you haven't passed `--send`. Every email it actually sends gets recorded in a local file so a later reply can be matched back to it.

**Phase 2** (`reply`): whenever someone replies, paste the reply in and give it the lead's id. It reads the reply, decides what the person meant (interested, not interested, pushing back on something, out of office, or asking to be left alone), and acts accordingly. If the reply says "stop" in any way, the code -- not the model -- marks that person as done and refuses to contact them again, no matter what the model itself suggests doing. That rule also carries forward: if that same person shows up in a future list of leads under a different id, this agent still won't email them again.

## How to run locally

```bash
git clone https://github.com/rajeshm71/real-world-agents.git
cd real-world-agents
cp .env.example .env    # then edit .env: set LLM_PROVIDER + the matching API key
cd agents/18_sales_outreach
```

Everything below runs as a dry run by default -- it shows you what it would do without sending anything or touching your email account. Add `--send` only once you want it to actually send.

```bash
uv run python -m agent send --leads examples/leads.json --sender examples/sender.json --product examples/product.md --icp examples/icp.md
```

Actually send (needs SMTP set up first -- see Setup below):

```bash
uv run python -m agent send --leads examples/leads.json --sender examples/sender.json --product examples/product.md --icp examples/icp.md --send
```

When a reply comes in, paste it into a text file and process it:

```bash
uv run python -m agent reply --lead-id acme_jdoe --reply-file reply.txt
```

Mock mode (no API key, canned response, for testing the pipeline end-to-end):

```bash
uv run python -m agent send --leads examples/leads.json --sender examples/sender.json --product examples/product.md --icp examples/icp.md --provider mock
```

### Setup: sending real email

This agent sends through a regular email account using an app password, not a separate email-sending service. With Gmail (the default):

1. Turn on 2-factor authentication on your Google account, if it isn't already.
2. Go to https://myaccount.google.com/apppasswords and generate an app password (free, takes a minute).
3. Set two environment variables: `SMTP_USER` (your Gmail address) and `SMTP_APP_PASSWORD` (the 16-character password you just generated).

Using a different provider? Set `SMTP_HOST` and `SMTP_PORT` to override the `smtp.gmail.com:587` default.

**Important**: every email this agent sends through `--send` includes a plain opt-out line and your postal address, drafted automatically. U.S. commercial-email law (CAN-SPAM) requires both on every commercial email, with real consequences for skipping them -- this agent won't let a draft through without them, but you are the one responsible for putting your real address in `sender.json`. And once someone replies asking to stop, this agent will never email them again, in this run or any future one -- that part isn't optional or model-decided, it's enforced in code.

## Code walkthrough

Under 500 LOC in `agent.py` (`smtp_client.py` and `state_store.py` hold the two pieces of real-world plumbing separately, each independently testable):

1. **`schemas.py`**: every shape that moves through the pipeline. `SenderProfile` carries the required postal address; `LeadOutreachRecord.outcome` is a closed list of 5 values so a reader can see every possible result of one lead without reading `agent.py`.
2. **`prompts/qualify.txt`**: asks for a 0-100 fit score and reasoning grounded in the actual lead details, not a vague guess.
3. **`prompts/draft_email.txt`**: asks for a short, personalized email, and requires the opt-out line and postal address to appear verbatim -- a draft missing either gets rejected and re-drafted (see `_check_draft` in `agent.py`), the same mechanism that checks the personalization is real and not invented.
4. **`agent.py::_build_phase1_graph()`**: the pedagogical anchor. Qualify, then a conditional branch (draft+send, or skip), with the real side effect living in `send_email_node`. `_route_after_qualify` is a plain function you can test without building the graph at all.
5. **`agent.py::run_outreach_batch()`**: the public entry point for phase 1. Checks for empty/bad input before anything else (R5 case 1), resolves SMTP credentials once up front rather than discovering a missing setup partway through a batch, checks each lead against the list of people who already asked to stop (by email, not by id, since the same person can show up under a different id in a later file), and if any lead fails partway through, keeps the results already completed rather than throwing them away.
6. **`agent.py::process_reply()`**: phase 2. Classifies the reply, then applies the code-owned stop rule before anything the model suggested gets a say.
7. **`smtp_client.py::_send_via_smtp()`**: the one function that opens a real connection. It never raises for a failure that's expected to happen sometimes (wrong password, connection refused, timeout) -- those come back as an ordinary result the rest of the code reads like any other, the same way `agent.py::_translate_api_error()` (R5 case 3, six branches: by exception type, by status code, by message text, an Ollama-specific hint, then a catch-all) handles failures from the LLM calls.
8. **`state_store.py`**: the file that bridges phase 1 and phase 2. Nothing else in this catalog reads its own output file back in a later run -- this is the one agent that does.
9. **`tests/test_smoke.py`**: 49 tests under `LLM_PROVIDER=mock`. The most important one scripts a reply classified as "not interested" where the model's own suggestion is to email again anyway, and checks that the code ignores that suggestion completely.

## When to use / When NOT to use

**Use when:**
- You have a real, specific list of leads (not a scraped mass list) and want personalized outreach, not a mail-merge template
- You're comfortable sending through your own email account at a low volume
- You want a record of what was sent and a clear, enforced rule for what happens after someone says stop

**Do NOT use when:**
- You need your inbox watched automatically -- replies have to be pasted in by hand in this version
- You're sending in bulk: this is a reasoning-and-drafting layer over a regular email account, not deliverability infrastructure (no dedicated sending domain, no bounce handling, no warm-up)
- Your list wasn't collected with real consent to be contacted -- this agent helps you write and send better email, it doesn't make the underlying list defensible

## Where this fails

- **No live inbox watching.** Someone has to notice the reply and paste it in. Real inbox polling needs IMAP or OAuth access to a live mailbox -- a bigger, different piece of infrastructure than this agent takes on, and the kind of thing this whole catalog avoids for agents that need a live account connected just to try them out.
- **A generic lead blurb produces a generic email.** This agent can't invent specifics about a company it wasn't told. The personalization check only confirms the email used a detail from the blurb -- it can't confirm the blurb itself was worth using.
- **The reply classifier can get it wrong.** Five categories can't cover every real reply -- sarcasm, a forwarded message in someone else's voice, or a reply that's genuinely ambiguous. The "stop" rule exists specifically because this is the one mistake that matters: a false "interested" just wastes a follow-up, but a missed "please stop" is the one this agent is built to never make, which is why that decision is made in code instead of left to the model's judgment.
- **No sending-rate limit built in.** Gmail caps regular accounts around 500 sends a day; this agent doesn't track that for you, and pushing against it risks your account getting flagged, not just an error message.
- **This agent drafts the required opt-out line and postal address into every email, but it can't verify what you put in `sender.json` is accurate.** That part is on you.
