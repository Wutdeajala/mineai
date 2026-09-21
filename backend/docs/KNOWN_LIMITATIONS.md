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