# Report writer, iteration 2: SAM vs SGD calibration on Fashion-MNIST

This workspace holds the internal research report for the run, as a chronological lab notebook.
The report is `paper_draft.md`. It keeps the iteration 1 record, adds the iteration 2 section,
and rewrites the closing section "What we have learned so far".

## Layout

- `paper_draft.md`: the full report. Iteration 1 and its corrections, iteration 2, the closing section and the reference list.
- `references.bib`, `references.json`: BibTeX for the six outside works cited, fetched through the Semantic Scholar tool. `references.json` is the record the tool writes beside the `.bib`.
- `figures/`: chart specs (`fig3_spec.json`, `fig4_spec.json`, `fig5_spec.json`) and their renders (`fig3_v1.png`/`.pdf`, `fig4_v0.png`/`.pdf`, `fig5_v1.png`/`.pdf`). Figure 1 and Figure 2 are from iteration 1 and are not re-rendered here.
- `tools/gen_tables.py`: reads the artifact output files and prints the iteration 2 tables as markdown, with the recomputed contrasts. It reads absolute paths under the run's `iter_1/gen_art` and `iter_2/gen_art` folders, which are read-only for this step.
- `domain_terms.json`, `domain_vocabulary.json`, `domain_corpus/`, `style_exemplars.md`, `field_extras.json`: the field vocabulary, the corpus it was built from, the style exemplars and the list of extras the field's best papers deliver.

## What the iteration did

Three artifacts, read in full in this step:

- Artifact 3 (art_v2M5ZOOHpgid): the headline comparison re-run with learning rates tuned on validation NLL over a grid that extends below the old edge.
- Artifact 4 (art_7yu4oYgjmi55): SAM at 40 epochs against SGD at matched training loss, with a temperature-scaled control.
- Evaluation 1 (art_V0gOM__yRf8b): the calibration gap read along two existing training trajectories, reported separately.

## Reproducing the tables

Run `python3 tools/gen_tables.py` from any folder that can read the artifact workspaces. It writes its markdown next to the script. The figures are rendered with the `aii-data-fig-gen` skill, from `figures/*_spec.json`:

    python3 <skill>/scripts/chart_gen.py --spec figures/fig3_spec.json --out figures/fig3_v1

## Restoring removed files

No file in this workspace is marked for deletion, so there is nothing to restore.
