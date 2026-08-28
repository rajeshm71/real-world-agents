# Example JD + resumes

Self-authored fake job description plus four synthetic candidate
resumes. No real people, no real companies. MIT alongside the rest of
the project.

- `job_description.md` -- "Senior Backend Engineer at a fintech
  startup." Realistic scope: Python, distributed systems, on-call,
  remote-first. Written from scratch.
- `resumes/alice.pdf` -- strong candidate (all boxes ticked). Built
  once from `scripts/build_example_resumes.py` (reportlab; run
  locally, not part of the workspace deps). The generator script is
  in the repo for reproducibility; regeneration only produces a
  binary diff so it's not part of CI.
- `resumes/bob.docx` -- borderline candidate (Django-only, no
  distributed systems). Built by the same helper via python-docx
  (already a runtime dep of this agent).
- `resumes/carol.md` -- "yes"-tier candidate written directly as
  markdown.
- `resumes/dave.md` -- "no"-tier candidate written directly as
  markdown.

Point `--resumes` at any of these files (or your own PDFs / DOCXs / MD
/ TXT files anywhere on disk) to try the screener.
