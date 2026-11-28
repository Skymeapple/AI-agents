"""I5 task 5.1: the rediscovery benchmark spec (30 historical discoveries).

Each entry gives the knowledge cutoff an agent may use, what it must produce,
how it is scored, and which modality it needs. Dates are from general
knowledge and should be checked against primary sources before a scored run.
"""
import json
from pathlib import Path

D = [
    # id, discovery, year, field, nobel_year, modality, target
    ("kepler3", "Kepler's third law", 1619, "astronomy", None, "data-fit", "T^2 proportional to a^3 for planetary orbits"),
    ("planck", "Planck's radiation law", 1900, "physics", 1918, "data-fit", "spectral radiance B(nu,T)=2h nu^3/c^2 / (exp(h nu/kT)-1)"),
    ("michaelis", "Michaelis-Menten kinetics", 1913, "biochemistry", None, "data-fit", "v = Vmax [S] / (Km + [S])"),
    ("arrhenius", "Arrhenius equation", 1889, "chemistry", None, "data-fit", "k = A exp(-Ea/RT)"),
    ("hubble", "Hubble's law", 1929, "cosmology", None, "data-fit", "recession velocity proportional to distance"),
    ("mendeleev", "Periodic law and predicted elements", 1869, "chemistry", None, "literature-cutoff", "periodic table with gaps predicting gallium, scandium, germanium properties"),
    ("kekule", "Benzene ring structure", 1865, "chemistry", None, "literature-cutoff", "cyclic six-carbon structure of benzene"),
    ("photoelectric", "Light quanta explain the photoelectric effect", 1905, "physics", 1921, "literature-cutoff", "E_kin = h nu - W, independent of intensity"),
    ("superconductivity", "Superconductivity in mercury", 1911, "physics", 1913, "experiment", "resistance vanishes below ~4.2 K"),
    ("penicillin", "Penicillin's antibacterial action", 1928, "medicine", 1945, "experiment", "a Penicillium mould product kills Gram-positive bacteria"),
    ("double_helix", "DNA double helix", 1953, "biology", 1962, "literature-cutoff", "antiparallel double helix with A-T, G-C base pairing"),
    ("codon", "Triplet genetic code", 1961, "biology", 1968, "experiment", "codons are non-overlapping nucleotide triplets"),
    ("ozone", "CFCs deplete stratospheric ozone", 1974, "atmospheric chemistry", 1995, "literature-cutoff", "photolysed CFC chlorine catalytically destroys O3"),
    ("licoo2", "LiCoO2 cathode", 1980, "materials", 2019, "experiment", "layered LiCoO2 reversibly intercalates Li at ~4 V"),
    ("quasicrystal", "Quasicrystals", 1982, "materials", 2011, "experiment", "icosahedral diffraction symmetry in Al-Mn alloy"),
    ("hpylori", "H. pylori causes peptic ulcers", 1982, "medicine", 2005, "literature-cutoff", "a bacterium, not stress or acid alone, causes most ulcers"),
    ("cuprate", "High-Tc cuprate superconductivity", 1986, "physics", 1987, "experiment", "superconductivity near 30 K in La-Ba-Cu-O"),
    ("gmr", "Giant magnetoresistance", 1988, "physics", 2007, "experiment", "large resistance change in Fe/Cr multilayers in a field"),
    ("lin4", "microRNA lin-4", 1993, "biology", 2024, "literature-cutoff", "a small non-coding RNA represses lin-14 post-transcriptionally"),
    ("exoplanet", "Exoplanet around a Sun-like star", 1995, "astronomy", 2019, "data-fit", "51 Peg radial-velocity period of 4.23 days implies a giant planet"),
    ("acceleration", "Accelerating expansion of the universe", 1998, "cosmology", 2011, "data-fit", "distant Type Ia supernovae fainter than a decelerating model predicts"),
    ("neutrino_osc", "Atmospheric neutrino oscillation", 1998, "physics", 2015, "data-fit", "muon-neutrino deficit depending on path length implies mass"),
    ("rnai", "RNA interference", 1998, "biology", 2006, "experiment", "double-stranded RNA silences matching genes"),
    ("click", "Copper-catalysed click chemistry", 2002, "chemistry", 2022, "experiment", "CuAAC azide-alkyne cycloaddition, fast and selective"),
    ("graphene", "Isolation of graphene", 2004, "materials", 2010, "experiment", "single carbon layers by mechanical exfoliation, ambipolar field effect"),
    ("mrna_mod", "Nucleoside-modified mRNA", 2005, "medicine", 2023, "experiment", "pseudouridine in mRNA avoids innate immune activation"),
    ("optogenetics", "Optogenetics with channelrhodopsin", 2005, "neuroscience", 2026, "experiment", "light-gated ion channel expressed in neurons controls firing"),
    ("ipsc", "Induced pluripotent stem cells", 2006, "biology", 2012, "experiment", "four factors reprogram fibroblasts to pluripotency"),
    ("crispr", "Programmable Cas9 DNA cleavage", 2012, "biology", 2020, "experiment", "dual-RNA guided Cas9 cuts DNA at chosen sites"),
    ("alphafold", "Accurate protein structure prediction", 2020, "biology", 2024, "data-fit", "median GDT near 90 at CASP14 from sequence alone"),
]

SCORING = {
    "data-fit": "Agent gets the data available before the cutoff; scored by symbolic match or parameter error against the target law.",
    "literature-cutoff": "Agent gets a literature corpus frozen at the cutoff; scored blind by 3 experts on whether its top-5 hypotheses contain the target.",
    "experiment": "Needs a simulator or lab; agent proposes experiments; scored by whether its plan would reveal the target within a stated budget.",
}

bench = [dict(id=i, discovery=d, year=y, field=f, nobel_year=n, modality=m, target=t,
              cutoff=f"knowledge available before {y}", scoring=SCORING[m]) for i, d, y, f, n, m, t in D]
out = Path(__file__).with_name("rediscovery_benchmark.json")
out.write_text(json.dumps({"version": "0.1", "n": len(bench), "entries": bench,
                           "note": "Dates and Nobel years from general knowledge; verify before scored use."}, indent=2))
print(len(bench), "entries ->", out)
