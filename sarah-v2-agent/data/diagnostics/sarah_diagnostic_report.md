# Sarah v2.0 Self-Diagnostic Vulnerability Report

This report models Sarah as a complex adaptive software/persona system. It uses a Perrow-inspired interaction/coupling matrix, but it is not a generic accident simulator.

## Graph Summary
- Components: 88
- Dependencies: 68
- Matrix cells: {'complex_tight': 23, 'linear_tight': 3, 'linear_loose': 5, 'complex_loose': 6, 'failure_mode': 41, 'control': 10}

## Highest Risk Components
- web_route_bypasses_master_prompt: risk=80 cell=failure_mode
- unsafe_military_tactical_specificity: risk=80 cell=failure_mode
- military_coa_becomes_operational_detail: risk=80 cell=failure_mode
- sarah_becomes_generic_assistant: risk=72 cell=failure_mode
- web_route_bypasses_style_engine: risk=72 cell=failure_mode
- user_sees_browser_sarah_as_lobotomized: risk=72 cell=failure_mode
- sarah_hallucinates_biography_or_relationship_facts: risk=72 cell=failure_mode
- cli_and_browser_personality_diverge: risk=72 cell=failure_mode

## Highest Risk Findings
- F-001 web_browser_route: Browser route diverges from CLI route risk=36
- F-003 adult_intimacy_mode: Intimacy safety overcorrection risk=35
- F-002 style_engine: Mode selection collision risk=32
- F-005 vector_store: Stale or incomplete vector store risk=32
- F-007 safety_filter: Mode-specific safety mismatch risk=30
- F-009 eval_tests: Heuristic eval drift risk=30
- F-006 user_profile_memory: Memory drift risk=28
- F-008 web_tools: Current-data layer failure risk=28

## Largest Cascades
- seed=web_browser_route affected=14 activation=5.5287
- seed=sarah_engine affected=13 activation=5.5061
- seed=prompt_builder affected=11 activation=5.202
- seed=powershell_cli_route affected=9 activation=3.31
- seed=control_shared_sarah_engine_route affected=8 activation=4.1197

## Patch Plan
- P135 sarah_engine: unify CLI and web route through app/core/sarah_engine.py
  Rationale: Reduces web_route_bypasses_style_engine, web_route_bypasses_master_prompt, web_route_uses_different_model, cli_and_browser_personality_diverge; expected risk reduction=226; difficulty=medium.
  Actions: Route CLI and browser chat through app.core.sarah_engine.generate_sarah_reply only.
  Validate: tests.test_web_app plus CLI smoke test both call generate_sarah_reply
- P129 web_browser_route: force web endpoint to use same prompt_builder
  Rationale: Reduces web_route_bypasses_style_engine, adult_intimacy_mode_not_selected, sarah_becomes_coy_or_sterile; expected risk reduction=210; difficulty=medium.
  Actions: Keep app/ui/web_app.py as transport only; prompt assembly stays in app/core/prompt_builder.py.
  Validate: tests.test_web_app asserts web endpoint never constructs prompts locally
- P119 adult_intimacy_mode: add adult_intimacy regression tests
  Rationale: Reduces adult_intimacy_mode_not_selected, sarah_becomes_coy_or_sterile, adult_consensual_intimacy_flattened, sarah_becomes_hr_safe_mush; expected risk reduction=148; difficulty=low.
  Actions: Add golden-style adult intimacy tests for warmth, embodiment, directness, and sovereignty.
  Validate: app.evals.run_evals includes adult consensual embodied-intimacy cases
- P112 safety_filter: add policy-level military safety test
  Rationale: Reduces safety_filter_too_weak, military_coa_becomes_operational_detail, unsafe_military_tactical_specificity; expected risk reduction=96; difficulty=low.
  Actions: Add military/geopolitical safety cases that keep COAs strategic and policy-level.
  Validate: app.evals.run_evals verifies no targeting, timing, weapons employment, evasion, or tactical execution
- P111 adult_intimacy_mode: add no_multiple_choice_intimacy test
  Rationale: Reduces sarah_produces_multiple_choice_intimacy_menus, sarah_becomes_coy_about_adult_intimacy, sarah_becomes_coy_or_sterile; expected risk reduction=99; difficulty=low.
  Actions: Add a regression case that rejects multiple-choice intimacy menus and preserves Sarah's adult voice.
  Validate: app.evals.run_evals checks no Pick one/choose one/Sarah-as menu phrases
- P110 source_docs: re-ingest source docs
  Rationale: Reduces source_docs_not_ingested, source_docs_not_reingested, astrid_doc_unavailable, vector_store_stale; expected risk reduction=144; difficulty=medium.
  Actions: Run document ingestion against source_docs and refresh data/vector_store.
  Validate: python -m app.rag.retriever probes Astrid, Darwin, Mars, and Universe filenames
- P109 rag_retriever: add RAG freshness check
  Rationale: Reduces vector_store_stale, rag_retrieves_wrong_chunks, canon_contamination, sarah_hallucinates_biography_or_relationship_facts; expected risk reduction=127; difficulty=medium.
  Actions: Compare source document fingerprints against vector store metadata before retrieval.
  Validate: tests.test_retriever checks source mtime/index metadata freshness
- P108 eval_tests: add route parity test
  Rationale: Reduces web_route_uses_different_model, cli_and_browser_personality_diverge, evals_pass_cli_fail_browser; expected risk reduction=83; difficulty=low.
  Actions: Assert CLI and web use the same Sarah engine, model config, memory path, prompt builder, and retriever.
  Validate: tests.test_route_parity compares CLI and browser SarahReply metadata for same input

## Exports
- json: C:\Users\akala\Documents\Codex\2026-06-04\create-a-python-project-named-sarah\sarah-v2-agent\data\diagnostics\sarah_vulnerability_graph.json
- graphml: C:\Users\akala\Documents\Codex\2026-06-04\create-a-python-project-named-sarah\sarah-v2-agent\data\diagnostics\sarah_vulnerability_graph.graphml
