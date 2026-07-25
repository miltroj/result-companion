# Plugin Direction — Honest Scope

**Goal**: keep Result Companion valuable in a world where general LLMs already read logs. Invest where RC still has a moat.

## Rule of thumb

If a user can paste raw output into Copilot / Claude and get the same answer in under a minute, RC adds nothing. RC earns its place only when input is too big, too noisy, or too framework-specific to hand to a general LLM as-is.

## Reality check (2026)

| Force | Effect on RC |
|---|---|
| Context windows (200k–2M) | Chunking value depreciating |
| Copilot / Bugbot / Claude PR review | PR-diff × failure cross-ref becoming commodity |
| Multimodal LLMs | Screenshots + traces now analyzable — nobody wires this to test reports yet |
| JUnit XML is small and flat | Paste-to-chat already works |
| Robot `output.xml` is huge, XML-shaped, rebot-fused | Paste-to-chat still fails — RC uniquely useful |

## Where RC uniquely wins

- **Robot `output.xml` parsing** — multi-MB, rebot fusion, depth-aware breadcrumbs, dedup, token-aware chunking. Real moat.
- **Robot screenshots → multimodal LLM** (unexplored). SeleniumLibrary / BrowserLibrary embed screenshots in `log.html`. Nobody feeds them to vision LLMs alongside the failure breadcrumb. Biggest untapped differentiator.
- **`log.html` enrichment** — RC lives inside the file QA teams already open daily.

## Scope (honest)

| Bucket | In / Out | Priority | Reason |
|---|---|---|---|
| Robot Framework core | in (existing) | keep polishing | real moat |
| **Robot image analysis** | **in — new** | **first** | no framework does this natively; direct moat extension |
| Universal JUnit XML | escape hatch only | second, minimal | ~200 LOC courtesy for mixed Robot + pytest teams; do not market as "15 frameworks" |
| Playwright native JSON | **deferred** | after vision pipeline proven | text-only version strictly worse than Playwright HTML + trace viewer |
| Unit-test only, pytest json, TestNG / Cypress native | out | — | short output OR JUnit already suffices |

## What actually transfers to non-Robot frameworks

| Value bucket | Robot users | JUnit users | Reason |
|---|---|---|---|
| Per-failure LLM analysis | 100% | ~20% | small JUnit failure fits one paste |
| PR review integration | 100% | ~30% | Copilot / Bugbot already do it |
| HTML enrichment | 100% | ~40% | GH Actions summary + Allure exist |
| Token-aware chunking | 100% | ~10% | JUnit rarely exceeds context |
| Image analysis (planned) | 100% | 0% | JUnit strips artifacts |

Old doc claimed "70-80% transfers." True for Robot-shaped inputs. False for JUnit / Playwright users' baseline.

## Priorities

1. **Robot image analysis prototype** — extract embedded screenshots from `log.html`, feed multimodal LLM with the failure breadcrumb, surface vision output in `rc_log.html` next to per-test analysis. Ship as opt-in flag first.
2. **JUnit escape hatch** — ~200 LOC `ParserPlugin` conformer. Ship quietly. No per-framework marketing. Positioned as "works if you already have Robot + one JUnit source."
3. **Playwright** — revisit only after Robot vision pipeline proves out. Reuse the pipeline against traces + screenshots then. Not before.

## Non-goals (say no on purpose)

- Multi-framework aggregator like Allure. Differentiator = LLM + vision, not report prettiness.
- Native adapters where framework's JUnit output already suffices (TestNG, Cypress).
- Text-only Playwright plugin — strictly worse than Playwright's own report.
- Competing head-on with Copilot / Bugbot as sole feature. Position RC as the *context builder* that makes any review agent better, not another review agent.
- Community plugin ecosystem push. Ship what we ship ourselves.
- Plugin infra beyond one `ParserPlugin` Protocol until a real third plugin demands it.

## Done when

- [ ] Robot image-analysis prototype: extract screenshots from `log.html`, call multimodal LLM, render vision output in `rc_log.html` next to text analysis.
- [ ] `ParserPlugin` Protocol lives in core; Robot plugin conforms.
- [ ] JUnit adapter ships as escape hatch; tested on real pytest output; not featured in README hero.
- [ ] `analyze --format` errors clearly when user asks for a capability the plugin does not support.
- [ ] Docs explain when RC beats "paste to Copilot Chat" and when it does not.

## Not doing yet

- Playwright native plugin (deferred until vision pipeline proven on Robot).
- `Capabilities` dataclass, `api_version`, conformance kit, renderer / reporter planes.
- Cucumber / Behave / Gauge native adapters.

## One-liner for README

> Result Companion — LLM + vision analysis for Robot Framework failures. Turns `output.xml` and screenshots into actionable failure summaries. JUnit XML supported as an escape hatch for mixed suites.
