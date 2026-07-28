"""Does the foreach (multi_tensor) path break at DeBERTa-scale tensor lists?

Mimics deberta-v3-base's parameter shapes (198 tensors, incl. a 128100x768
embedding) and checks both AdamW and clip_grad_norm_ under
foreach / single-tensor implementations.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "8")
import torch

torch.manual_seed(0)


def deberta_like_shapes():
    s = [(128100, 768), (512, 768)]          # word emb + rel emb
    for _ in range(12):                      # 12 layers
        s += [(768, 768), (768,)] * 4        # q,k,v,o
        s += [(768,), (768,)]                # attn LN
        s += [(3072, 768), (3072,)]          # ffn in
        s += [(768, 3072), (768,)]           # ffn out
        s += [(768,), (768,)]                # out LN
    s += [(768, 768), (768,), (1, 768), (1,)]  # pooler + head
    return s


def build(shapes, gscale=0.01):
    ps = []
    for sh in shapes:
        p = torch.randn(*sh, device="cuda", dtype=torch.float32,
                        requires_grad=True)
        p.grad = torch.randn_like(p) * gscale
        ps.append(p)
    return ps


shapes = deberta_like_shapes()
print(f"{len(shapes)} tensors, "
      f"{sum(int(torch.tensor(s).prod()) for s in shapes)/1e6:.1f}M params")

# ---- clip_grad_norm_ : foreach vs single ----
for fe in (True, False):
    torch.manual_seed(0)
    ps = build(shapes)
    ref = torch.sqrt(sum((p.grad.double() ** 2).sum() for p in ps))
    n = torch.nn.utils.clip_grad_norm_(ps, 1.0, foreach=fe)
    after = torch.sqrt(sum((p.grad.double() ** 2).sum() for p in ps))
    print(f"clip foreach={fe}: reported={n.item():.6f} ref={ref.item():.6f} "
          f"norm_after_clip={after.item():.6f} (expect ~1.0) "
          f"nan={bool(torch.stack([torch.isnan(p.grad).any() for p in ps]).any())}")
    del ps
    torch.cuda.empty_cache()

# ---- AdamW step at scale ----
for impl, kw in [("fused", {"fused": True}),
                 ("foreach", {"foreach": True}),
                 ("single", {"foreach": False, "fused": False})]:
    torch.manual_seed(0)
    ps = build(shapes)
    opt = torch.optim.AdamW(ps, lr=2e-5, weight_decay=0.01, **kw)
    bad = []
    for step in range(3):
        opt.step()
        nan_t = sum(int(torch.isnan(p).any()) for p in ps)
        if nan_t:
            bad.append((step, nan_t))
        for p in ps:
            p.grad = torch.randn_like(p) * 0.01
    print(f"AdamW {impl:8s}: nan_events={bad if bad else 'none'}")
    del ps, opt
    torch.cuda.empty_cache()
