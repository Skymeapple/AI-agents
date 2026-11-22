import csv, os, numpy as np, datetime as dt
HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, '..', 'data'); R = os.path.join(HERE, '..', 'results')
os.makedirs(R, exist_ok=True)

LAB_ALIAS = {
 'Google': 'Alphabet', 'Google DeepMind': 'Alphabet', 'DeepMind': 'Alphabet', 'Google Research': 'Alphabet',
 'Google Brain': 'Alphabet', 'Meta AI': 'Meta', 'Facebook AI Research': 'Meta', 'Facebook AI': 'Meta',
 'Facebook': 'Meta', 'Microsoft Research': 'Microsoft', 'Microsoft Research Asia': 'Microsoft',
 'Z.ai (Zhipu AI)': 'Zhipu', "Huawei Noah's Ark Lab": 'Huawei', 'Alibaba Cloud': 'Alibaba',
}
# Capital-bloc attribution (sensitivity analysis): labs mapped to their largest strategic investor
# as publicly reported. This is deliberately coarse; see paper Sec. 4.3.
BLOC_ALIAS = dict(LAB_ALIAS)
BLOC_ALIAS.update({'OpenAI': 'Microsoft', 'Anthropic': 'Amazon', 'Inflection AI': 'Microsoft', 'Mistral AI': 'Mistral AI'})

OPEN = {'Open weights (unrestricted)', 'Open weights (restricted use)', 'Open weights (non-commercial)'}

def todate(s):
    return dt.date.fromisoformat(s)

def year_frac(d):
    return d.year + (d.timetuple().tm_yday - 1) / 365.25

def actor(org_field, mode='lab'):
    alias = LAB_ALIAS if mode == 'lab' else BLOC_ALIAS
    first = org_field.split(',')[0].strip()
    a = alias.get(first, first)
    if mode == 'bloc':
        a = BLOC_ALIAS.get(a, a)
    return a

def load(min_date='2010-01-01'):
    rows = []
    for r in csv.DictReader(open(os.path.join(D, 'notable_parsed.csv'))):
        if r['Publication date'] < min_date: continue
        c = r['Training compute (FLOP)']
        cost = r['Training compute cost (2023 USD)']
        rows.append(dict(model=r['Model'], org=r['Organization'], date=todate(r['Publication date']),
                         t=year_frac(todate(r['Publication date'])),
                         C=float(c) if c else None, cost=float(cost) if cost else None,
                         access=r['Model accessibility'], open=r['Model accessibility'] in OPEN,
                         country=r['Country (of organization)'], cat=r['Organization categorization'],
                         epoch_frontier=r['Frontier model'] == 'True', domain=r['Domain']))
    rows.sort(key=lambda x: x['t'])
    return rows

def hhi(counts):
    v = np.array(list(counts.values()), float)
    if v.sum() == 0: return np.nan
    s = v / v.sum(); return float((s ** 2).sum())
