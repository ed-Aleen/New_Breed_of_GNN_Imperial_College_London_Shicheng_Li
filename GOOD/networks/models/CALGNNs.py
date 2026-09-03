r"""
CAL (Causal Attention Learning, Sui et al., KDD 2022, arXiv:2112.15089) adapted to
this codebase's node-attention SE-GNN skeleton.

WHY THIS MODEL EXISTS HERE (experiment E1, family dial): CAL is the Omega == 0 family —
its training objective contains NO term that is a function of the attention scores alone
(no KL-to-r, no L1, no entropy). The three CAL losses are all data terms:
    L_sup  = CE(z_causal, y)
    L_unif = KL(softmax(z_trivial) || uniform)          [paper Eq. 4 region]
    L_caus = CE(classifier(rep_c + rep_t[perm]), y)     [paper Eq. 5, random intervention]

Faithful-adaptation notes (documented deviations from the original implementation):
  * Scores: the paper uses a 2-way softmax (alpha_c, alpha_t) per node; here the extractor
    emits ONE logit l_i and alpha_c = sigmoid(l_i). This is EXACTLY the same function
    (2-way softmax == sigmoid of the logit difference).
  * Masking: causal branch masks with alpha_c (node mask + product-lifted edge mask via
    lift_node_att_to_edge_att), trivial branch with 1 - alpha_c — same masking infra as
    GSAT/SMGNN in this repo, replacing the paper's separate-branch convolutions with the
    shared encoder + separate classifier heads. The shared-vs-split encoder is NOT
    load-bearing for the Omega == 0 property.
  * Intervention: representation-level random addition within the batch, gradients flow
    into both branches (the paper's equation states no stop-gradient).
  * Sampling: DETERMINISTIC always — CAL has no sampling noise, in training or eval.

Config contract (see configs/final_configs/MUTAG/basis/no_shift/CAL.yaml):
    model.model_name: CAL      ood.ood_alg: CAL
    ood.extra_param = [learn_edge_att(False), lambda_unif, lambda_caus]
"""
import torch

from GOOD import register
from GOOD.utils.config_reader import Union, CommonArgs, Munch
from GOOD.utils.train import lift_node_att_to_edge_att
from .Classifiers import Classifier
from .GSATGNNs import GSAT, set_masks, clear_masks


@register.model_register
class CAL(GSAT):

    def __init__(self, config: Union[CommonArgs, Munch]):
        super(CAL, self).__init__(config)
        # separate trivial-branch classifier head (paper: separate trivial readout/clf)
        self.classifierT = Classifier(config, is_linear=False)
        self.trivial_logits = None
        self.interv_logits = None
        assert not self.learn_edge_att, "CAL adaptation assumes node-level attention"
        print("Using CAL (deterministic attention, Omega == 0: no score regulariser)")

    def sampling(self, att_log_logits, training, mitigation_expl_scores):
        # CAL is deterministic: no concrete/Gumbel noise ever (training included).
        return att_log_logits.sigmoid()

    def forward(self, *args, **kwargs):
        data = kwargs.get('data')

        emb = self.gnn(*args, without_readout=True, **kwargs)
        att_log_logits = self.extractor(emb, data.edge_index, data.batch)
        att = self.sampling(att_log_logits, self.training, self.config.mitigation_expl_scores)

        # ---------------- causal branch ----------------
        edge_att_c = lift_node_att_to_edge_att(att, data.edge_index)
        set_masks(edge_att_c, self, att)
        rep_c = self.gnn(*args, **kwargs)              # pooled graph representation
        clear_masks(self)
        logits_c = self.classifierS(rep_c)

        # ---------------- trivial branch ----------------
        att_t = 1.0 - att
        edge_att_t = lift_node_att_to_edge_att(att_t, data.edge_index)
        set_masks(edge_att_t, self, att_t)
        rep_t = self.gnn(*args, **kwargs)
        clear_masks(self)
        self.trivial_logits = self.classifierT(rep_t)

        # ---------------- random intervention (paper Eq. 5) ----------------
        perm = torch.randperm(rep_t.shape[0], device=rep_t.device)
        self.interv_logits = self.classifierS(rep_c + rep_t[perm])

        self.edge_mask = edge_att_c
        return logits_c, att_log_logits, att
