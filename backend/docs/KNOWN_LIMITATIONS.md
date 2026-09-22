# Known Limitations — Tracked for Future Work

## Contradiction Detection (Phase 10)
Built and tested across two full fix iterations (sentence-aware chunking, table
separation). The NLI model (cross-encoder/nli-deberta-v3-small) still over-predicts
"contradiction" on dense academic/legal prose, even on clean, complete, genuinely
compatible sentences. Currently NOT wired into the app. See contradiction.py for
full details. Next step: try a larger or legal-domain-tuned NLI model, or an
LLM-based comparison approach.

## Geographic/Categorical Scope Conflation (Phase 10)
When evidence describes something generally (e.g., minerals found across Nigeria)
without specifically confirming it for a narrower entity asked about (e.g., Plateau
State specifically), the model can present the general statement as if it directly
answers the specific question. A prompt-level fix was attempted and confirmed NOT
to reliably work via testing. Next step: try constraining retrieval itself using
hybrid search (keyword + embeddings) so exact-location matches are favored over
general topical similarity - not yet implemented.

## Risk/Gap Detection Relevance Filtering (Phase 10)
check-gaps and check-risks now filter retrieved chunks to distance < 0.5 before
running analysis, fixing an earlier issue where loosely-related chunks produced
off-topic gap/risk statements. This threshold is a reasonable default, not
empirically tuned.

## Multi-Page Table Extraction (Phase 10)
Tables spanning multiple PDF pages without a repeated header can have column
values shift/misalign (see ingest.py table extraction). Single-page tables
extract cleanly. Not yet fixed - affects table chunk_type content quality on
long tables specifically.

## Reranker Default (Phase 6)
Built, integrated as a config flag (use_reranker), tested against a formal
evaluation set - found no measurable improvement (and slight MRR regression)
on the current small corpus. Defaults to OFF. Worth re-testing once corpus
is significantly larger.


## External Search Citation Integration (Phase 9)
When automatic external search triggers (see external_search.py, rag.py ask()),
web results are included in the generation prompt and correctly labeled as
"EXTERNAL WEB SOURCE (unverified)" - the model does treat them with appropriate
caution in its answers (tested: correctly refused to state an unverified gold
price as fact). However, external sources are NOT currently integrated into the
structured citation/grounding system (check_grounding only knows about local
database chunks) - so external claims appear in the answer text but don't show
up in the clickable citations panel. A user can't verify an external claim the
same way they can a local one. Next step: extend check_grounding to also accept
and verify against external source snippets.

## External Search Trigger Logic (Phase 9)
Initial trigger only checked evidence_status (insufficient_evidence/
partially_supported) - found via testing that a well-grounded answer correctly
stating "no local data exists" scores as "supported" (the statement itself IS
supported), so the trigger never fired. Fixed by adding a second, heuristic
keyword check on the answer text itself (phrases like "no information",
"does not provide", etc.) alongside the evidence_status check. This is a blunt
fix, not a robust one - could false-positive/negative on differently-phrased
gaps. Works for now; revisit if it proves unreliable in practice.