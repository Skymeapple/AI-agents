"""BLS Employment Projections, 'Occupational separations and openings, projected 2025-35'
(table at https://www.bls.gov/emp/tables/occupational-separations-and-openings.htm, accessed 2026-11-22).
Columns transcribed verbatim: employment 2025 (thousands), employment 2035 (thousands),
labor-force exit rate, occupational transfer rate, total occupational separations rate
(annual averages, percent)."""

BLS = [
    # name, soc, emp2025, emp2035, exit, transfer, total_sep
    ("Software developers",                "15-1252", 1717.8, 1892.6, 1.4, 2.9, 4.3),
    ("Computer programmers",               "15-1251",  110.8,  102.7, 1.9, 3.0, 4.9),
    ("Software QA analysts and testers",   "15-1253",  187.6,  198.2, 1.8, 3.2, 5.0),
    ("Web developers",                     "15-1254",   87.4,   90.6, 2.0, 3.6, 5.6),
    ("Accountants and auditors",           "13-2011", 1595.2, 1674.6, 2.7, 3.9, 6.6),
    ("Financial and investment analysts",  "13-2051",  377.2,  404.3, 1.9, 3.7, 5.6),
    ("Market research analysts",           "13-1161",  952.7, 1019.0, 2.5, 5.1, 7.6),
    ("Lawyers",                            "23-1011",  863.7,  904.3, 1.6, 1.2, 2.8),
    ("Paralegals and legal assistants",    "23-2011",  404.9,  403.8, 3.0, 7.1, 10.1),
    ("Customer service representatives",   "43-4051", 2666.0, 2524.1, 5.0, 6.7, 11.7),
    ("Registered nurses",                  "29-1141", 3465.4, 3660.1, 2.5, 2.0, 4.5),
]
