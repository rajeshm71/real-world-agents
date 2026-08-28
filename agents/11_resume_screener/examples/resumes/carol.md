# Carol Chen

Backend engineer, 6 years of Python in production. Currently at
Wefund, previously at NorthPay.

## Experience

**Wefund -- Senior Backend Engineer** (2024 -- present)

- Built the reconciliation service that matches provider payouts
  against our internal ledger. Runs nightly across ~40M rows.
- Sole author of the async fraud-signal ingestor: consumes from an
  SQS queue at ~200 messages/second, writes to Postgres with
  exactly-once semantics using an idempotency key + unique index.
- Rotated on-call for the payment path (one week every four); wrote
  the runbook for the "provider webhook backlog" playbook after an
  incident where we caught up 14K delayed events in 30 minutes.

**NorthPay -- Backend Engineer** (2021 -- 2024)

- Owned the merchant-facing REST APIs (Python 3.10, FastAPI, Postgres
  15). Refactored the historical webhooks handler to be idempotent
  after a duplicate-charge incident.
- Contributed to the deposit-hold service that held funds during ACH
  clearing. Learned double-entry accounting the hard way.

## Skills

Python (6y), Postgres, SQS, async/await, idempotency patterns,
FastAPI, pytest, Terraform (basic), Kafka (evaluated, did not ship).

## Working style

Fully remote for the last three years across four time zones. Prefer
async decisions in written docs, but happy to hop on a call for
anything ambiguous.
