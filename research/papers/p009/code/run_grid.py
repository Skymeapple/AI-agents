"""Run experiment grids in parallel (4 single-threaded workers). Skips runs whose JSON exists.
python run_grid.py main|state|seeds3"""
import itertools, os, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
OUT = os.path.join(os.path.dirname(__file__), '..', 'results', 'runs')
COMMON = ['--k', '2', '--N', '32', '--steps', '4000', '--lr', '3e-3']

def job(layers, seed, dk=4, extra=()):
    name = f"{layers}_k2_N32_dk{dk}_s{seed}.json"
    if os.path.exists(os.path.join(OUT, name)):
        return
    cmd = [sys.executable, 'hops.py', '--layers', layers, '--seed', str(seed), '--dk', str(dk), '--out', OUT] + COMMON + list(extra)
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=os.path.dirname(os.path.abspath(__file__)))
    print(r.stdout.strip() or r.stderr[-500:], flush=True)

GRID4 = ['DDDD', 'ADDD', 'DADD', 'DDAD', 'DDDA', 'AADD', 'ADAD', 'ADDA', 'DAAD', 'DADA', 'DDAA', 'AAAA', 'A', 'AA']
which = sys.argv[1]
jobs = []
if which == 'main':
    jobs = [(l, s, 4) for s in (0, 1) for l in GRID4]
elif which == 'seeds3':
    jobs = [(l, 2, 4) for l in GRID4]
elif which == 'state':
    jobs = [(l, s, dk) for s in (0, 1) for dk in (2, 8, 16, 32) for l in ('DDDD', 'DDDA', 'ADDD')]
with ThreadPoolExecutor(4) as ex:
    list(ex.map(lambda j: job(*j), jobs))
