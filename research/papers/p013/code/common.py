"""Shared utilities: random-fact memorization with an explicit abstain class.
Keys are random D-bit strings (universe size 2^D, D=32 => effectively open world);
values are uniform on [V]. Class V = ABSTAIN ("I don't know")."""
import math, numpy as np, torch
D = 32

def make_net(h, V, depth=2, head='softmax'):
    """head='softmax': V+1 logits (V values + ABSTAIN).
    head='bits': 1 'answer' logit + log2(V) value-bit logits (parameter count ~independent of V)."""
    layers, d = [], D
    for _ in range(depth):
        layers += [torch.nn.Linear(d, h), torch.nn.ReLU()]; d = h
    out = V + 1 if head == 'softmax' else 1 + int(round(math.log2(V)))
    layers += [torch.nn.Linear(d, out)]
    return torch.nn.Sequential(*layers)

def nparams(net):
    return sum(p.numel() for p in net.parameters())

def rand_keys(m, gen=None):
    return (torch.randint(0, 2, (m, D), generator=gen) * 2 - 1).float()

def h2(p):
    if p <= 0 or p >= 1: return 0.0
    return -p * math.log2(p) - (1 - p) * math.log2(1 - p)

def log2binom(n, r):
    return (math.lgamma(n + 1) - math.lgamma(r + 1) - math.lgamma(n - r + 1)) / math.log(2)

def certified_bits(g, k, eps, V, n=2.0 ** D):
    """Theorem 1 lower bound (bits a model MUST contain) given measured recall g of k
    stored facts and false-acceptance rate eps on uniform fresh keys.
    B >= g log2 V + log2 C(n,g) - log2 C(abar,g) - log2 C(k,g),  abar = k + eps (n-k)."""
    if g <= 0: return 0.0
    abar = k + eps * (n - k)
    lb = g * math.log2(V) + log2binom(n, g) - log2binom(abar, g) - log2binom(k, g)
    return max(lb, 0.0)

def value_bits(y, H):
    return ((y.unsqueeze(1) >> torch.arange(H)) & 1).float()

def loss_fn(net, X, y, Xn, V, w, head='softmax'):
    if head == 'softmax':
        F = torch.nn.functional
        return F.cross_entropy(net(X), y) + w * F.cross_entropy(net(Xn), torch.full((len(Xn),), V))
    H = int(round(math.log2(V))); F = torch.nn.functional
    op, on = net(X), net(Xn)
    return (F.binary_cross_entropy_with_logits(op[:, 0], torch.ones(len(X)))
            + F.binary_cross_entropy_with_logits(op[:, 1:], value_bits(y, H), reduction='none').sum(1).mean()
            + w * F.binary_cross_entropy_with_logits(on[:, 0], torch.zeros(len(Xn))))

@torch.no_grad()
def predict(net, X, V, head='softmax'):
    if head == 'bits':
        o = net(X); conf = torch.sigmoid(o[:, 0])
        H = o.shape[1] - 1
        ans = ((o[:, 1:] > 0).long() << torch.arange(H)).sum(1)
        return conf.numpy(), ans.numpy()
    p = torch.softmax(net(X), dim=1)
    conf = 1.0 - p[:, V]                       # probability of NOT abstaining
    ans = p[:, :V].argmax(1)                   # value answered if it answers
    return conf.numpy(), ans.numpy()
