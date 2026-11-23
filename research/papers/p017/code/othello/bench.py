import time, torch, torch.nn.functional as F
from gpt import GPT
import sys; torch.set_num_threads(int(sys.argv[1]))
for d, nl in [(128, 4), (128, 6), (192, 4)]:
    m = GPT(d, nl); opt = torch.optim.AdamW(m.parameters(), 1e-3)
    x = torch.randint(0, 60, (128, 60))
    t0 = time.time()
    for _ in range(5):
        lg = m(x[:, :-1]); loss = F.cross_entropy(lg.reshape(-1, 61), x[:, 1:].reshape(-1)); opt.zero_grad(); loss.backward(); opt.step()
    dt = (time.time() - t0) / 5
    print(d, nl, sum(p.numel() for p in m.parameters()), f"{dt:.3f}s/step, {128*59/dt:.0f} tok/s")
