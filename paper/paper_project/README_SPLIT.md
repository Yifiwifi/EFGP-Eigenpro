# Split LaTeX project

Compile `main.tex`, not the files in `sections/` directly.

Recommended workflow:

1. Compile `main.tex` once with all `\includeonly` lines commented out.
2. Run BibTeX/LaTeX as usual until references resolve.
3. While editing one section, uncomment the matching `\includeonly{sections/...}` line in `main.tex`.
4. Before final submission, comment out all `\includeonly` lines and compile the full paper again.

Files:
- `sections/01-introduction.tex` — \section{Introduction}
- `sections/02-background.tex` — \section{Background: EFGP Weight-Space Kernel Regression and Eignepro Preconditioner}
- `sections/03-main-algorithms.tex` — \section{Main Algorithms: Structured Active-Set Preconditioners}
- `sections/04-implementation.tex` — \section{Algorithms and Implementation}
- `sections/05-experiments.tex` — \section{Numerical Experiments}
- `sections/06-discussion-limitations.tex` — \section{Discussion and Limitations}
- `sections/07-reproducibility.tex` — \section*{Reproducibility}
- `sections/08-effectiveness-analysis.tex` — \section{Effectiveness Analysis}

Notes:
- `references (3)(1).bib` was renamed to `references.bib` so `\bibliography{references}` works cleanly.
- The original self-recursive macro `\newcommand{\spec}{\spec}` was changed to `\DeclareMathOperator{\spec}{spec}`.
- A few matrix row separators that ended with a single backslash were changed to `\\`.
