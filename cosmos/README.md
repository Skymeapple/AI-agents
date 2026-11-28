# cosmos

Data and runnable simulations for ten open problems in cosmology. It lives beside
`sciscout` and follows the same rule: every number carries where it came from.

```
pip install -e ".[cosmos]"
python -m cosmos.run_all          # all ten, about 30 seconds
python -m cosmos.run_all 2 8      # selected problems
```

Outputs go to `cosmos/output/`: one figure per problem, `results.json` with every
computed number, and `REPORT.md`, which pairs each figure with what it shows and what
the simulation leaves out.

## What is simulated

| # | Problem | Simulation | Data it runs on |
|---|---|---|---|
| 1 | Dark matter | Rotation curve under three hypotheses; WIMP recoil rate in xenon | Synthetic galaxy; LZ limit and exposure |
| 2 | Dark energy | Fit of constant against evolving dark energy | DESI DR2 BAO, Pantheon+ supernovae, CMB prior |
| 3 | Hubble tension | Tension statistics; sound-horizon degeneracy; toy early dark energy | Nine H0 measurements; DESI DR2 BAO |
| 4 | Inflation | Numerical slow-roll predictions for five potentials | Tilt measurements; gravitational-wave limit |
| 5 | Matter over antimatter | Boltzmann equations for thermal leptogenesis | Baryon abundance; neutrino mass scale |
| 6 | Black-hole information | Entanglement entropy of random 14-qubit states (Page curve) | None needed |
| 7 | Early black holes | Accretion growth from three kinds of seed | Seven black holes at redshift 7 to 10.6 |
| 8 | Structure and dipole | Linear growth; Monte Carlo of the quasar-count dipole | S8 from six analyses; CatWISE dipole |
| 9 | Lithium | Neutron freeze-out and helium yield; abundance scalings | Observed helium, deuterium, lithium |
| 10 | Neutrino mass | Mass sums for both orderings against the limits | Oscillation splittings; DESI and KATRIN limits |

## The data

- `data/desi_dr2_bao_*.txt`: the DESI DR2 compressed BAO likelihood, unmodified, from
  the public `CobayaSampler/bao_data` repository.
- `data/pantheon_plus_binned.npz`: the 1590 Pantheon+ supernovae above redshift 0.01,
  compressed to 40 bins using the full statistical and systematic covariance.
  `build_sn_binned.py` regenerates it from the public release (the 33 MB covariance is
  not committed).
- `data/measurements.yaml`: every other number. Each entry is `sourced` (with a
  reference and a `checked` level) or `assumed` (with a rationale).

The `checked` level matters. `document` means the value was read in the paper or data
file. `search-summary` means it was seen in a search summary of the paper. `recalled`
means it was written from memory and has **not** been re-verified; most entries are in
this state, and they should be checked before anything is built on them.

## How far to trust it

The dark-energy fit is the one result here that can be held against a published
number, and it reproduces the DESI collaboration's chi-squared improvements for DESI
alone and DESI with the CMB prior to within 0.1, and for DESI with Pantheon+ to within
0.6. A test asserts this.

The rest are deliberately small models: they show the mechanism and the size of each
problem, and each states its own limits in `REPORT.md`. Three limits are worth knowing
up front:

- The dark-matter rotation curve uses a synthetic galaxy. The SPARC database host was
  not reachable from the environment this was built in.
- The helium yield from the freeze-out integration is about 9% below the observed
  value; a full reaction network is needed to do better.
- The toy early-dark-energy model needs a larger energy fraction than published fits,
  because it holds every other parameter fixed.
