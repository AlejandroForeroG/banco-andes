# Editorial reviewer

An independent service using HTTP POST and SSE events signed with HMAC-SHA256. It implements a custom application protocol, rather than the A2A standard.

`critic.py` queries versioned synthetic policies in SQLite and evaluates the assistant's answer. `server.py` validates requests and streams verdicts; `protocol.py` defines the contract and signatures. The database contains two policy versions, five criteria, and three examples. These are not real banking regulations.

Two parameterized, read-only SELECT queries retrieve policies. The service computes the verdict from criterion coverage and findings. The assistant verifies the signature, correlation, deadline, and consistency before showing validated text.

The cycle allows two critiques and one refinement, with a 90-second total budget and 30 seconds per model call. The second review reuses the policy snapshot and original deadline.

```bash
python -m reviewer.init_sources
python -m reviewer.query_sources --date 2026-09-23 --scope housing
python -m unittest reviewer.test_sources -v
```

Initialization does not overwrite an existing database. Use `--db PATH` for a new copy. An unavailable source produces `failed/SOURCE_UNAVAILABLE`.
