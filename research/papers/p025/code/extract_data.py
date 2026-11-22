"""Extract the human reference trajectories from a clone of WukLab/osworld-human.

Usage: python code/extract_data.py /path/to/osworld-human-clone
Writes data/oswh_tasks.json with, per task: id, app, instruction, single, grouped.
The upstream repository asks that its contents not be used for training; we use it
only for evaluation-time analysis (human step counts and action types).
"""
import glob, json, os, sys

def main(repo):
    out = []
    for f in sorted(glob.glob(os.path.join(repo, '*', '*.json'))):
        d = json.load(open(f))
        h = d.get('human-ground-truth')
        if not h:
            continue
        out.append(dict(id=d['id'], app=os.path.basename(os.path.dirname(f)),
                        instruction=d['instruction'],
                        single=h['single-action'], grouped=h['grouped-action']))
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, '..', 'data', 'oswh_tasks.json')
    json.dump(out, open(path, 'w'), indent=1)
    print(f'wrote {len(out)} tasks to {os.path.normpath(path)}')

if __name__ == '__main__':
    main(sys.argv[1])
