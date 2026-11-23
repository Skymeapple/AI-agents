"""Keystroke-Level-Model (KLM) estimate of expert, error-free human execution time
for each OSWorld-Human reference trajectory.

Operator values from Card, Moran & Newell (1980): P=1.1 s (point), B=0.1 s (button
press or release), H=0.4 s (home hands between mouse and keyboard), M=1.35 s (mental
preparation), K = 0.28 s (average non-secretary typist; 0.20 s for an average skilled
typist). Instruction reading uses 238 words/min (Brysbaert 2019, non-fiction silent
reading). SCROLL and WAIT have no canonical KLM value; we use explicit assumptions
(SCROLL_S, WAIT_S) and vary them in the sensitivity analysis.
"""
import re

P, B, H, M = 1.1, 0.1, 0.4, 1.35
DEFAULTS = dict(K=0.28, SCROLL_S=1.0, WAIT_S=2.0, M_MODE='grouped', READ_WPM=238.0)

MOUSE_ACTIONS = {
    'CLICK': P + 2 * B, 'RIGHT_CLICK': P + 2 * B, 'CLOSE_WINDOW': P + 2 * B,
    'SELECT': 2 * (P + 2 * B), 'CHECK': P + 2 * B,
    'DOUBLE_CLICK': P + 4 * B, 'DOUBLECLICK': P + 4 * B,
    'TRIPLE_CLICK': P + 6 * B,
    'MOVE_TO': P, 'MOVE_ON': P,
    'DRAG_TO': B + P + B, 'DRAG': B + P + B, 'DRAG_AND_DROP': P + B + P + B,
    'MOUSE_DOWN': P + B, 'MOUSE_UP': P + B,
}
KEY_ACTIONS = {'PRESS', 'RRESS', 'HOTKEY', 'HOTEKY', 'HOTEKEY', 'KEY_DOWN', 'KEY_UP',
               'KEYDOWN', 'KEYUP', 'KEY DOWN', 'KEY UP', 'TYPING', 'SHELL', 'OPEN_FILE'}


def parse(action):
    a = action.strip()
    m = re.match(r'`([A-Z_ ]+?)`\s*(.*)', a)
    if m:
        return m.group(1).strip(), m.group(2)
    m = re.match(r'`([A-Z_]+)\s*(.*?)`?$', a)  # e.g. "`MOVE_TO bottom right ...`"
    if m:
        return m.group(1), m.group(2)
    return 'CLICK', a


def typed_chars(arg):
    q = re.findall(r"'([^']*)'|\"([^\"]*)\"", arg)
    if q:
        return sum(len(x or y) for x, y in q)
    return max(len(arg.split(' in ')[0].strip()), 1)


def action_time(kind, arg, p, hand):
    """Return (seconds, new_hand_state, is_terminal_fail)."""
    if kind == 'FAIL':
        return 0.0, hand, True
    if kind in ('SCROLL', 'SCROLL_LEFT'):
        t = p['SCROLL_S']; need = 'mouse'
    elif kind == 'WAIT':
        return p['WAIT_S'], hand, False
    elif kind == 'REPEAT_UNTIL_DONE':
        # "for the remaining 4 emails": treat as 4 repetitions of a click+type+enter unit
        n = int((re.findall(r'(\d+)', arg) or ['1'])[0])
        return n * (M + P + 2 * B + H + 20 * p['K'] + p['K']), 'kbd', False
    elif kind in MOUSE_ACTIONS:
        t = MOUSE_ACTIONS[kind]; need = 'mouse'
    elif kind in KEY_ACTIONS:
        need = 'kbd'
        if kind in ('TYPING', 'SHELL', 'OPEN_FILE'):
            t = typed_chars(arg) * p['K'] + (p['K'] if kind != 'TYPING' else 0.0)
        elif kind in ('HOTKEY', 'HOTEKY', 'HOTEKEY'):
            keys = [k for k in re.split(r'[+\-]', arg.strip(" '\"")) if k.strip()]
            t = max(len(keys), 2) * p['K']
        else:
            t = p['K']
    else:
        t = MOUSE_ACTIONS['CLICK']; need = 'mouse'
    if hand != need and hand is not None:
        t += H
    return t, need, False


def klm_time(task, **kw):
    """Expert error-free execution time (s) of one OSWorld-Human task.

    M_MODE='grouped': one mental operator per grouped-action group (one decision per
    observation, which is how OSWorld-Human defines a group); 'single': one per action.
    Returns dict with total, motor, mental, reading seconds.
    """
    p = dict(DEFAULTS); p.update(kw)
    hand = 'mouse'
    motor = 0.0
    for a in task['single']:
        k, arg = parse(a)
        t, hand, _ = action_time(k, arg, p, hand)
        motor += t
    n_dec = len(task['grouped']) if p['M_MODE'] == 'grouped' else len(task['single'])
    mental = M * n_dec
    reading = 60.0 * len(task['instruction'].split()) / p['READ_WPM']
    return dict(total=motor + mental + reading, motor=motor, mental=mental, reading=reading,
                infeasible=any(parse(a)[0] == 'FAIL' for a in task['single']))
