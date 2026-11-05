QOTAM: full-interaction versus low-order perturbative comparison

Additional calculation performed for the manuscript discussion. This is not an
updated manuscript package and does not replace the 2026-10-10 revision.

Run:
  python compare_full_and_perturbative.py --output-dir results

Requires Python 3.10+, NumPy and SciPy. Tested versions are recorded in comparison.json.

Model (same parameters and finite field domain as the revised manuscript):
  S = sum_j phi_j^2/2 + (0.7/2)[(phi_2-phi_1)^2+(phi_3-phi_2)^2]
      + (lambda/24) sum_j phi_j^4,
  phi_j in [-6,6], hbar=1, lambda=0.5, 2, 4.

Full-interaction branch:
  n=80 positive Gauss-Legendre nodes per field; 100 entropic mirror updates,
  step size 1/4; no coupling truncation. The independent validation branch uses
  nested adaptive quadrature directly from the action, not the grid or optimizer.

Perturbative branch:
  Let U = sum_j phi_j^4/24. Expectation values are Taylor-expanded at lambda=0
  with the normalized ratio explicitly re-expanded. For observable O:
  O0 = <O>0
  O1 = -Cov0(O,U)
  O2 = Cov0(O,U^2)/2 - <U>0 Cov0(O,U).
  Connected fourth cumulants are re-expanded consistently through the same order.
  All expectation values and reference normalizations use the SAME field cutoff.
  Refining the perturbative coefficient quadrature from n=80 to n=96 is checked.

Files:
  compare_full_and_perturbative.py  standalone calculation
  comparison.csv                 all observables and relative differences
  comparison.json                coefficients, method settings, references, versions
  run.log                        executed results

Scope:
  This is a small three-site regulated Euclidean scalar calculation, not a
  three-spatial-dimensional field simulation or a Lorentzian scattering result.
  It demonstrates retaining the full interaction, not the impossibility of
  accurate higher-order/resummed perturbation theory. It is not an OT-specific
  speedup or general sign-problem result. Small cross-method differences are
  numerical agreement, not rigorous interval error bounds.
