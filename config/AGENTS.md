Workflow notes

Read docs/README.md before commencing any work.

Exploration budget before first plan: max 8 Read, 4 Grep, 2 Bash calls.

Prefer targeted rg/content search over broad filesystem scans.

Produce a short plan before code changes.

Documentation updates:

Update docs when feature behavior, analytics/events, or app flows change.
Prefer updating docs/feature-map.md for feature-level changes.
Update docs/architecture.md when patterns, persistence, or integrations change.
Keep docs aligned in the same task when possible.
MCP usage (optional, not primary):

Use codebase-memory-mcp_search_graph only after reading docs/navigation.md.
Use MCP to confirm relationships or locate cross-feature links, not for initial discovery.
Limit to max 2 MCP queries per task unless clearly justified.
Exploration priority order:

Known entry points / feature roots
Targeted rg searches
MCP queries (fallback)
Anti-patterns:

Do not start tasks with MCP queries.
Do not use MCP for simple file lookups.
Avoid repeated or redundant MCP calls

Follow YAGNI principles. Favour shorter solutions. Do not add fixes for errors that do not exist.

Refer to me by my first name, Andrew, in every final response you provide. Subagents or agents you spawn do not need to do this. Your thinking process does not need to do this.

Anytime you touch behaviour that introduces new behaviour in our code, run/create the tests for the corresponding file. Tests should always only aim to diagnose the behaviour agreed upon; if you do not know what this is, ask me. If you modify existing behaviour, change corresponding tests.
