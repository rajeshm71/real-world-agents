# Sample handbook

Five short Markdown files under `handbook/` mimicking a fake company HR
handbook (benefits, leave, code of conduct, remote work, expenses).
Used by the Gradio UI's sample-picker and by the tests as a realistic
corpus for ingestion and retrieval smoke tests.

All content is self-authored for this demo. No real company, no real
policies, no third-party rights involved. Numbers and specifics are
plausible-sounding but not tied to any actual employer's practices --
if you want to use these as a template for your own handbook, verify
every claim against your jurisdiction's employment law and your own
company's actual policies first.

If you want to demo the agent against your own docs, drop them into
any directory and pass its path to `python -m agent ingest <dir>` --
you don't have to use the shipped handbook.
