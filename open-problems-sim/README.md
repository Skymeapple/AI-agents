# Open-problem simulations

Numerical experiments for ten famous unsolved problems. **Nothing here proves
anything.** Each experiment reproduces the conjectured behaviour in a small
finite range and is checked against published values where they exist.

    pip install numpy scipy mpmath matplotlib
    python simulate.py            # all modules, about 10 minutes on 4 cores
    python simulate.py abc bsd    # selected modules
    python plot.py                # results.json -> figures.png

All data is generated locally from first principles; nothing is downloaded.

| # | Problem | What is simulated | Result | Check |
|---|---------|-------------------|--------|-------|
| 1 | Riemann hypothesis | First 150 zeros, spacing statistics, psi(x) - x up to 10^7, Li coefficients 1-16 | Mean spacing 0.994; max abs(psi(x)-x)/sqrt(x) = 0.92; all Li coefficients positive | First zero 14.1347 |
| 2 | P vs NP | Random 3-SAT at 50 variables with a DPLL solver; exact formula size of all 65,536 four-variable functions | Satisfiability drops from 100% to 0% around 4.3 clauses per variable, where search cost peaks; 4-bit parity needs 16 leaves | Khrapchenko bound n^2 = 16 |
| 3 | Navier-Stokes | 3D Taylor-Green vortex, pseudo-spectral, 64^3, Re = 100, 200, 400 | Peak enstrophy 0.65, 1.28, 2.22; peak vorticity 5.8, 11.4, 15.4 | Energy balance holds to 0.05% |
| 4 | Yang-Mills | SU(2) lattice gauge theory, 8^4, Metropolis | Plaquette follows beta/4 at strong coupling and crosses smoothly to weak coupling; Creutz ratio falls from 2.04 to 0.13 | Plaquette 0.567 at beta 2.2 (literature about 0.569) |
| 5 | Birch-Swinnerton-Dyer | Product of N_p/p for primes below 50,000 on curves of rank 0-3 | Fitted growth exponents -0.03, 1.06, 1.85, 3.34 | Conjecture predicts 0, 1, 2, 3 |
| 6 | Hodge | Hodge numbers of hypersurfaces; the 48 lines on the Fermat quartic | The lines span a lattice of rank 20 | h^(1,1) = 20 |
| 7 | Goldbach | Representation counts for every even n to 20,000 and 400 random n to 10^7 | Count / Hardy-Littlewood prediction between 0.981 and 1.016 | |
| 8 | Twin primes | Twin pairs below 10^7 | 58,980 against a prediction of 58,758 | Known count 58,980 |
| 9 | Collatz | Every start below 10^6; 20,000 random 60-bit starts | All reach 1; longest orbit 524 steps (837,799); 1.996 halvings per tripling | Known record 837,799 |
| 10 | abc | Every coprime a + b = c with rad(abc) < c, c up to 10^5 | 418 hits below 10^5; best quality 1.568 (1 + 4374 = 4375) | Known counts 6, 31, 120, 418 |

## Limits

- Hodge has no meaningful numerical simulation. The module tabulates where the
  conjecture is non-trivial and checks one classical case where it is a theorem.
- The Navier-Stokes run at Re = 400 is only marginally resolved at 64^3, and
  three Reynolds numbers say nothing about the limit of vanishing viscosity.
- The Yang-Mills run measures no mass gap. It shows the absence of a phase
  transition between the strong-coupling regime, where a gap is proven, and
  weak coupling.
- The BSD exponents are noisy; the product converges like a power of log X.
