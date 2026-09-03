r"""
Loss side of the CAL adaptation (see GOOD/networks/models/CALGNNs.py for the model and
for the faithful-adaptation notes). Total loss:

    L = CE(z_causal, y) + lambda_unif * KL(trivial || uniform) + lambda_caus * CE(z_interv, y)

There is deliberately NO term that is a function of the attention scores alone
(Omega == 0 family — experiment E1's fourth regulariser arm).

Config: ood.extra_param = [learn_edge_att, lambda_unif, lambda_caus]; ood.ood_param unused (set 0).
"""
import torch
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import Batch

from GOOD import register
from GOOD.utils.config_reader import Union, CommonArgs, Munch
from GOOD.utils.initial import reset_random_seed
from GOOD.utils.train import at_stage
from .BaseOOD import BaseOODAlg


def _uniform_kl(logits: Tensor) -> Tensor:
    r"""KL(pred || uniform), supporting both multi-logit (softmax) and single-logit
    (Bernoulli) heads. For softmax: KL = log C - H(p). For a single logit z:
    KL(Bern(sigmoid(z)) || Bern(0.5))."""
    if logits.shape[-1] > 1:
        logp = F.log_softmax(logits, dim=-1)
        p = logp.exp()
        C = logits.shape[-1]
        return (p * (logp + torch.log(torch.tensor(float(C), device=logits.device)))).sum(-1).mean()
    z = logits.view(-1)
    p = torch.sigmoid(z).clamp(1e-6, 1 - 1e-6)
    return (p * torch.log(2 * p) + (1 - p) * torch.log(2 * (1 - p))).mean()


def _ce(logits: Tensor, y: Tensor) -> Tensor:
    if logits.shape[-1] > 1:
        return F.cross_entropy(logits, y.long().view(-1))
    return F.binary_cross_entropy_with_logits(logits.view(-1), y.float().view(-1))


@register.ood_alg_register
class CAL(BaseOODAlg):

    def __init__(self, config: Union[CommonArgs, Munch]):
        super(CAL, self).__init__(config)
        self.att = None
        self.edge_att = None
        self.lambda_unif = float(config.ood.extra_param[1])
        self.lambda_caus = float(config.ood.extra_param[2])

    def stage_control(self, config: Union[CommonArgs, Munch]):
        if self.stage == 0 and at_stage(1, config):
            reset_random_seed(config)
            self.stage = 1

    def output_postprocess(self, model_output: Tensor, **kwargs) -> Tensor:
        raw_out, self.att, self.edge_att = model_output
        return raw_out

    def loss_postprocess(self, loss: Tensor, data: Batch, mask: Tensor,
                         config: Union[CommonArgs, Munch], epoch: int = None, **kwargs) -> Tensor:
        self.mean_loss = loss.mean()
        unif = _uniform_kl(self.model.trivial_logits)
        caus = _ce(self.model.interv_logits, data.y)
        self.spec_loss = self.lambda_unif * unif + self.lambda_caus * caus
        self.total_loss = self.mean_loss + self.spec_loss
        return self.total_loss
