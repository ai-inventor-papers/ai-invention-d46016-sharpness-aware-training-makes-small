# SAM versus SGD calibration on a small CNN (Fashion-MNIST), report workspace

This workspace holds the internal research report for iteration 1 of the run: the question, two artifacts (a controlled comparison and a mechanism study), their tables and figures, and the references and field vocabulary used to write the report.

## Layout

- `paper_draft.md`: the report. Framing, then the iteration 1 section, then the planned-but-not-run record, then the references.
- `references.bib`, `references.json`: bibliography fetched from Semantic Scholar, with a record for each entry.
- `style_exemplars.md`: verbatim passages and section headings from five field papers, used for register.
- `domain_terms.json`: field terms extracted from those papers (69 entries).
- `domain_vocabulary.json`, `domain_corpus/`: vocabulary built from full texts of five papers (about 27k words), with `sources.json`.
- `field_extras.json`: extras that the field's papers deliver beyond the minimum, as seen in those papers.
- `figures/fig1_spec.json`, `figures/fig2_spec.json`, and the rendered `fig1_v0`, `fig2_v0` (PNG and PDF): the two charts in the report, rendered from the artifacts' numbers with the data-figure renderer.

## Reproducing the figures

```bash
uv run --no-project --python 3.12 --with matplotlib --with numpy --with scipy python <aii-data-fig-gen>/scripts/chart_gen.py --spec figures/fig1_spec.json --out figures/fig1_v0
uv run --no-project --python 3.12 --with matplotlib --with numpy --with scipy python <aii-data-fig-gen>/scripts/chart_gen.py --spec figures/fig2_spec.json --out figures/fig2_v0
```

## Source artifacts

The numbers come from two artifacts in the run's workspace on the run's storage volume, not from this folder:

- Controlled comparison: `gen_art_experiment_1` (artifact id `art_SL9gPfE09HlN`). Its code, per-run tables and paired statistics are in that artifact's folder.
- Mechanism study: `gen_art_experiment_2` (artifact id `art_WOFinDmS2PoX`). Its per-epoch record, temperature-scaling record and matched-loss record are in its `results/` folder.

Model checkpoints and logits are kept on the run's volume and are not part of this folder.

## Restoring removed files

Nothing in this folder is marked for deletion. The heavy artifacts (checkpoints, logits) sit in the artifact folders on the run's volume, and this folder holds only text, bibliography, and small figure files.
