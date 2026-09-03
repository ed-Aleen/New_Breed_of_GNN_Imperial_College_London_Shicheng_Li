"""Gradient-capable masked classifier pass (port of the research repo helper).

The models' predict_from_subgraph is decorated @torch.no_grad(); this helper
replicates its body exactly (set_masks -> classifierS(gnn_clf or gnn) ->
clear_masks, resolved from the model's own module) without the decorator and
returns RAW LOGITS. Used by occ_stage_a.py / occ_channel_check.py.
"""
import sys

import torch
import torch.nn.functional as F


def masked_logits(model, ood_algorithm, batch, node_att):
    mod = sys.modules[model.__class__.__module__]
    mod.set_masks(False, model, node_att)
    try:
        if getattr(model, "gnn_clf", None):
            logits = model.classifierS(model.gnn_clf(
                data=batch, edge_weight=None, ood_algorithm=ood_algorithm))
        else:
            logits = model.classifierS(model.gnn(
                data=batch, edge_weight=None, ood_algorithm=ood_algorithm))
    finally:
        mod.clear_masks(model)
    return logits


def nll_from_logits(logits, ys, reduction="mean"):
    """ys: labels; logits [B,C] or [B,1] (single sigmoid logit)."""
    if logits.shape[-1] == 1:
        return F.binary_cross_entropy_with_logits(
            logits.view(-1), ys.float().view(-1), reduction=reduction)
    return F.cross_entropy(logits, ys.long().view(-1), reduction=reduction)


def probs_from_logits(logits):
    return logits.softmax(1) if logits.shape[-1] > 1 else logits.sigmoid()


def sanity_check(model, ood_algorithm, batch, node_att):
    """Assert this helper reproduces predict_from_subgraph's probabilities."""
    with torch.no_grad():
        lg = masked_logits(model, ood_algorithm, batch, node_att)
        p1 = lg.softmax(1) if lg.shape[-1] > 1 else lg.sigmoid()
        p2 = model.predict_from_subgraph(
            data=batch, edge_weight=None, edge_attn=None,
            node_att=node_att, ood_algorithm=ood_algorithm)
    err = (p1 - p2).abs().max().item()
    assert err < 1e-5, f"masked_logits mismatch vs predict_from_subgraph: {err}"
    return err
