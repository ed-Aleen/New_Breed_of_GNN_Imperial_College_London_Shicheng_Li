import torch, glob
for f in sorted(glob.glob("storage/expl_shapes/*.pt")):
    d = torch.load(f)
    m = d["meta"]
    top = sorted(d["dishes"].items(), key=lambda kv: -kv[1])[:6]
    n = sum(d["dishes"].values())
    print(f"{m['dataset']}-{m['model']} seed{m['seed']}: " + " ; ".join(f"{k}x{v}({100*v/n:.0f}%)" for k, v in top))
