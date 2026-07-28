"""Isolate AdamW correctness on this torch build: synthetic params, known grads.

Compares fused / foreach / single-tensor torch AdamW against a hand-written
reference Adam update. Any NaN or large deviation on step 1 is a torch bug,
not a model/data problem.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "8")
import torch

torch.manual_seed(0)


def make_params(n=5, shape=(64, 64), dtype=torch.float32):
    ps = [torch.randn(*shape, device="cuda", dtype=dtype, requires_grad=True)
          for _ in range(n)]
    for p in ps:
        p.grad = torch.randn_like(p) * 0.01
    return ps


def reference_step(p0, g, lr, wd, b1, b2, eps):
    # AdamW, first step: m=(1-b1)g, v=(1-b2)g^2, bias-corrected
    m = (1 - b1) * g
    v = (1 - b2) * g * g
    mh = m / (1 - b1)
    vh = v / (1 - b2)
    return p0 * (1 - lr * wd) - lr * mh / (vh.sqrt() + eps)


LR, WD, B1, B2, EPS = 2e-5, 0.01, 0.9, 0.999, 1e-8

for impl, kw in [("fused", {"fused": True}),
                 ("foreach", {"foreach": True}),
                 ("single", {"foreach": False, "fused": False})]:
    torch.manual_seed(0)
    ps = make_params()
    p0 = ps[0].detach().clone()
    g0 = ps[0].grad.detach().clone()
    opt = torch.optim.AdamW(ps, lr=LR, weight_decay=WD, betas=(B1, B2), eps=EPS,
                            **kw)
    opt.step()
    ref = reference_step(p0, g0, LR, WD, B1, B2, EPS)
    got = ps[0].detach()
    nan_any = any(bool(torch.isnan(p).any()) for p in ps)
    maxdiff = (got - ref).abs().max().item()
    print(f"{impl:8s} nan={nan_any}  max|got-ref|={maxdiff:.3e}  "
          f"got[0,:3]={got[0,:3].tolist()}  ref[0,:3]={ref[0,:3].tolist()}")

# Does it depend on multiple param groups / non-contiguous / param count?
print("\n-- single param, foreach --")
torch.manual_seed(0)
p = torch.randn(8, 8, device="cuda", requires_grad=True)
p.grad = torch.randn_like(p) * 0.01
opt = torch.optim.AdamW([p], lr=LR, foreach=True)
opt.step()
print("nan:", bool(torch.isnan(p).any()), p.detach()[0, :3].tolist())

print("\n-- torch._foreach_ ops sanity --")
a = [torch.ones(4, 4, device="cuda") for _ in range(3)]
b = [torch.full((4, 4), 2.0, device="cuda") for _ in range(3)]
r = torch._foreach_add(a, b, alpha=0.5)
print("foreach_add ->", r[0][0, :3].tolist(), "(expect [2,2,2])")
r2 = torch._foreach_addcdiv(a, b, b)
print("foreach_addcdiv ->", r2[0][0, :3].tolist(), "(expect [2,2,2])")
r3 = torch._foreach_sqrt(b)
print("foreach_sqrt ->", r3[0][0, :3].tolist(), "(expect ~1.414)")
r4 = torch._foreach_lerp(a, b, 0.5)
print("foreach_lerp ->", r4[0][0, :3].tolist(), "(expect [1.5,1.5,1.5])")
r5 = torch._foreach_mul(a, [torch.tensor(2.0, device="cuda")] * 3)
print("foreach_mul(scalar tensor) ->", r5[0][0, :3].tolist(), "(expect [2,2,2])")
