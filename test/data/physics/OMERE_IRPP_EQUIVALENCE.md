# OMERE IRPP equivalence controls

These spectra isolate numerical conventions without fitting a correction factor
to the Versal result.

Use the same one-cell Weibull component for every run:

- depth: `2 um`
- LET threshold: `2.57 MeV.cm2/mg`
- saturation cross-section: `5.721e-11 cm2/bit`
- width: `17.9`
- shape: `0.975`
- proton contribution: disabled

Run OMERE's SEE calculation once with each input:

1. `omere_irpp_uniform.let` gives equal differential weight from LET 1 to 4
   and primarily checks geometry normalization.
2. `omere_irpp_low_band.let` concentrates the flux below LET 2 and checks
   threshold, chord-tail, and lower-bin handling.
3. `omere_irpp_high_band.let` concentrates the flux above LET 2 and checks
   upper-bin and endpoint handling.

Record the unrounded rate displayed by OMERE when available and retain each
generated `.see` file. Compare the three rates separately with the framework;
do not derive or apply a global multiplier from them.

Interpretation:

- A common ratio in all three cases indicates geometry or solid-angle
  normalization.
- A low-band-only difference indicates chord-tail or threshold handling.
- A high-band-only difference indicates LET interpolation or upper-endpoint
  handling.
- Agreement for controls but disagreement for the Versal spectrum indicates
  hidden precision in OMERE's fitted Weibull parameters.

The production framework uses the ECSS/Petersen differential-spectrum
integral with continuous Weibull response. Alternative exported-grid
conventions are calculated only as diagnostics.
