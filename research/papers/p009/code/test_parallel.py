"""Check that the parallel delta-rule form equals the sequential recurrence."""
import torch; from hops import Delta
torch.manual_seed(0)
for delta in (True, False):
    m = Delta(64, 4, 8, delta=delta).double(); x = torch.randn(3, 40, 64).double()
    y1 = m(x); m.loop = True; y2 = m(x)
    print('delta' if delta else 'linear', 'max abs diff', (y1 - y2).abs().max().item())
