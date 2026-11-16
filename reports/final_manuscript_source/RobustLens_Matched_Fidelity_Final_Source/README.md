# RobustLens matched-fidelity manuscript source

This package contains the final two-column LaTeX manuscript, its compiled PDF, and all six vector PDF figures required to reproduce the paper.

## Files
- `RobustLens_Matched_Fidelity_Manuscript_Final.tex` — editable LaTeX source.
- `RobustLens_Matched_Fidelity_Manuscript_Final.pdf` — compiled manuscript.
- `figures/fig1_design_pipeline.pdf` through `figures/fig6_local_ssim_coefficients.pdf` — manuscript figures.

## Compile
Run twice from this directory:

```bash
pdflatex -interaction=nonstopmode -halt-on-error RobustLens_Matched_Fidelity_Manuscript_Final.tex
pdflatex -interaction=nonstopmode -halt-on-error RobustLens_Matched_Fidelity_Manuscript_Final.tex
```

The bibliography is embedded in the LaTeX source (`thebibliography` environment); no separate BibTeX file is required.

The manuscript is an unpublished research draft. Authors should approve author order, affiliations, data/code availability URLs, and scientific claims before submission.
