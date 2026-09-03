r"""OCC (Occurrence-Calibrated Classifier) utilities. PORTS.md experiment addition.

Vectorised hash-based broadcast-enriched colour refinement, the mu0
class-level random-mask sampler (REAL-X Bern(r0) masks lifted to colour-class
granularity), and a true induced-subgraph rebuilder used by the
channel-equivalence certificate (occ_channel_check.py).
"""
import torch
from torch_scatter import scatter_add

_A = 6364136223846793005   # odd LCG multiplier; int64 wrap-around hashing
_B = 1442695040888963407


def _hash(t):
    return t * _A + _B


def initial_colour_keys(x, decimals=4):
    """int64 hash key per node from (quantised) input features."""
    if x.dim() == 1:
        x = x.unsqueeze(1)
    xq = torch.round(x.float() * (10 ** decimals)).to(torch.int64)
    d = xq.shape[1]
    mult = _hash(torch.arange(1, d + 1, dtype=torch.int64, device=x.device))
    return (xq * mult).sum(1)


def refine_colour_keys(keys, edge_index, batch, rounds=2, enrich=True):
    """Broadcast-enriched colour refinement on hash keys.

    One round: key' = 7*H(own) + 31*sum_{u in N(v)} H(key_u)
                      [+ 131*sum_{u in G} H(key_u) if enrich]
    The neighbour term is a commutative multiset hash; the graph term is the
    ACR broadcast enrichment (vanilla WL misses the isolated-node splits, E1).
    Keys are deterministic functions of the graph -> comparable across graphs
    and batches. Collisions are hash-level (~2^-60 per pair), not systematic.
    """
    src, dst = edge_index[0], edge_index[1]
    nb = int(batch.max()) + 1 if batch.numel() else 0
    for _ in range(int(rounds)):
        h = _hash(keys)
        neigh = torch.zeros_like(keys).index_add_(0, dst, h[src])
        key = h * 7 + neigh * 31
        if enrich:
            gsum = scatter_add(h, batch, dim=0, dim_size=nb)
            key = key + gsum[batch] * 131
        keys = key
    return keys


def node_class_keys(data, granularity="class", rounds=2, decimals=4):
    """Per-node colour-class key for a (batched) Data object."""
    if granularity == "node":
        return torch.arange(data.x.shape[0], dtype=torch.int64,
                            device=data.x.device)
    keys = initial_colour_keys(data.x, decimals=decimals)
    return refine_colour_keys(keys, data.edge_index, data.batch, rounds=rounds)


def detect_granularity(x):
    """'class' for exactly one-hot rows (discrete attributes), else 'node'."""
    if x.dim() < 2:
        return "node"
    binary = bool(((x == 0) | (x == 1)).all())
    onehot = binary and bool((x.sum(1) == 1).all())
    return "class" if onehot else "node"


def sample_class_mask(keys, batch, r0=0.5, generator=None):
    """mu0: independent Bern(r0) per realised (graph, colour-class); returns a
    float {0,1} node mask conditioned on a non-empty display per graph
    (D' convention: an all-off graph gets one uniformly-random class on)."""
    device = keys.device
    pair = torch.stack([batch.to(torch.int64), keys], 0)
    uniq, inv = torch.unique(pair, dim=1, return_inverse=True)
    ngroup = uniq.shape[1]
    bern = (torch.rand(ngroup, generator=generator) < r0).to(device)
    num_graphs = int(batch.max()) + 1
    grp_graph = uniq[0]
    kept = torch.zeros(num_graphs, dtype=torch.long, device=device)
    kept.index_add_(0, grp_graph, bern.long())
    empty = (kept == 0).nonzero(as_tuple=True)[0]
    if len(empty):
        scores = torch.rand(ngroup, generator=generator).to(device)
        for g in empty.tolist():
            gsel = (grp_graph == g).nonzero(as_tuple=True)[0]
            bern[gsel[torch.argmax(scores[gsel])]] = True
    return bern[inv].float()


def induced_subbatch(data, keep):
    """True subgraph re-encoding: rebuild (x, edge_index, batch) keeping only
    `keep` (bool per node). Every graph must keep >= 1 node."""
    from torch_geometric.data import Data
    from torch_geometric.utils import subgraph
    ei, _ = subgraph(keep, data.edge_index, relabel_nodes=True,
                     num_nodes=data.x.shape[0])
    sub = Data(x=data.x[keep], edge_index=ei)
    sub.batch = data.batch[keep]
    sub.y = data.y
    return sub
