# Imported coding & web skills

Skills live in `.claude/skills/<name>/SKILL.md` and are loaded automatically by Claude Code.
Third-party licenses are in `third_party_licenses/`; per-skill `LICENSE.txt` files are kept where the source had them.

| Source | License | Skills |
|---|---|---|
| [anthropics/skills](https://github.com/anthropics/skills) | per-skill `LICENSE.txt` (Apache-2.0 for these) | frontend-design, webapp-testing, mcp-builder, skill-creator, claude-api, web-artifacts-builder |
| [vercel-labs/agent-skills](https://github.com/vercel-labs/agent-skills) | no LICENSE file found in repo root — verify before redistributing | react-best-practices, composition-patterns, web-design-guidelines, react-view-transitions |
| [supabase/agent-skills](https://github.com/supabase/agent-skills) | MIT | supabase-postgres-best-practices |
| [obra/superpowers](https://github.com/obra/superpowers) | MIT | systematic-debugging, test-driven-development, verification-before-completion, requesting-code-review, receiving-code-review, writing-plans |
| [wshobson/agents](https://github.com/wshobson/agents) | MIT | Python (16: async, typing, testing, packaging, uv, …), JS/TS (4), backend/API (api-design-principles, microservices, CQRS, saga, …), postgresql-table-design, fastapi-templates, Next.js app router, React state, Tailwind, React Native, CI/CD (GitHub Actions, GitLab CI), Kubernetes (helm, manifests, gitops, security), secrets-management |

Not imported: document skills (pdf/docx/pptx/xlsx — source-available, not code-related) and trailofbits/skills (CC BY-SA 4.0).
Each skill was skimmed for pipe-to-shell / obfuscated commands; none found. Re-review before trusting any script.
