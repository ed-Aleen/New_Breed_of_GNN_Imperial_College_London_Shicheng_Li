"""
Base class for OOD algorithms
"""
from abc import ABC
from torch import Tensor
from torch_geometric.data import Batch
from torch_scatter import scatter_sum
from torch_scatter.composite import scatter_softmax

from GOOD.utils.config_reader import Union, CommonArgs, Munch
from typing import Tuple
from GOOD.utils.initial import reset_random_seed
from GOOD.utils.train import at_stage
import torch



class BaseOODAlg(ABC):
    r"""
    Base class for OOD algorithms

        Args:
            config (Union[CommonArgs, Munch]): munchified dictionary of args
    """

    def __init__(self, config: Union[CommonArgs, Munch]):
        super(BaseOODAlg, self).__init__()
        self.optimizer: torch.optim.Adam = None
        self.scheduler: torch.optim.lr_scheduler._LRScheduler = None
        self.model: torch.nn.Module = None
        self.config = config

        self.mean_loss = None
        self.spec_loss = None
        self.stage = 0

        # Placeholder variables
        self.l_norm_loss = torch.nan #torch.tensor(0)
        self.entr_loss = torch.nan #torch.tensor(0)

    def stage_control(self, config):
        r"""
        Set valuables before each epoch. Largely used for controlling multi-stage training and epoch related parameter
        settings.

        Args:
            config: munchified dictionary of args.

        """
        if self.stage == 0 and at_stage(1, config):
            reset_random_seed(config)
            self.stage = 1

    def input_preprocess(self,
                         data: Batch,
                         targets: Tensor,
                         mask: Tensor,
                         node_norm: Tensor,
                         training: bool,
                         config: Union[CommonArgs, Munch],
                         **kwargs
                         ) -> Tuple[Batch, Tensor, Tensor, Tensor]:
        r"""
        Set input data format and preparations

        Args:
            data (Batch): input data
            targets (Tensor): input labels
            mask (Tensor): NAN masks for data formats
            node_norm (Tensor): node weights for normalization (for node prediction only)
            training (bool): whether the task is training
            config (Union[CommonArgs, Munch]): munchified dictionary of args

        Returns:
            - data (Batch) - Processed input data.
            - targets (Tensor) - Processed input labels.
            - mask (Tensor) - Processed NAN masks for data formats.
            - node_norm (Tensor) - Processed node weights for normalization.

        """
        return data, targets, mask, node_norm

    def output_postprocess(self, model_output: Tensor, **kwargs) -> Tensor:
        r"""
        Process the raw output of model

        Args:
            model_output (Tensor): model raw output

        Returns (Tensor):
            model raw predictions

        """
        return model_output

    def loss_calculate(self, raw_pred: Tensor, targets: Tensor, mask: Tensor, node_norm: Tensor,
                       config: Union[CommonArgs, Munch], batch: Tensor = None) -> Tensor:
        r"""
        Calculate loss

        Args:
            raw_pred (Tensor): model predictions
            targets (Tensor): input labels
            mask (Tensor): NAN masks for data formats
            node_norm (Tensor): node weights for normalization (for node prediction only)
            config (Union[CommonArgs, Munch]): munchified dictionary of args (:obj:`config.metric.loss_func()`, :obj:`config.model.model_level`)

        .. code-block:: python

            config = munchify({model: {model_level: str('graph')},
                                   metric: {loss_func: Accuracy}
                                   })


        Returns (Tensor):
            cross entropy loss

        """
        loss = config.metric.loss_func(raw_pred, targets, reduction='none') * mask
        loss = loss * node_norm * mask.sum() if config.model.model_level == 'node' else loss
        self.clf_loss = loss.detach().mean().item()   # UNWEIGHTED: keeps logs comparable
        w = drive_norm_weights(loss, targets, mask, config)
        if w is not None:
            loss = loss * w
            self.drive_norm_w_std = float(w.detach().std())
        return loss
    
    def loss_classifier(self, raw_pred: Tensor, targets: Tensor, mask: Tensor, node_norm: Tensor,
                       config: Union[CommonArgs, Munch], batch: Tensor = None) -> Tensor:
        loss = config.metric.loss_func(raw_pred, targets, reduction='none') * mask
        self.clf_loss = loss.detach().mean().item()
        return loss    

    def entropy_loss(logits, return_raw=False):
        logp = torch.log(logits + 0.0000000001)
        entropy = torch.sum(-logits * logp, dim=1)
        if not return_raw:
            entropy = torch.mean(entropy)
        return entropy

    def loss_postprocess(self, loss: Tensor, data: Batch, mask: Tensor, config: Union[CommonArgs, Munch], epoch:int,
                         **kwargs) -> Tensor:
        r"""
        Process loss

        Args:
            loss (Tensor): base loss between model predictions and input labels
            data (Batch): input data
            mask (Tensor): NAN masks for data formats
            config (Union[CommonArgs, Munch]): munchified dictionary of args

        Returns (Tensor):
            processed loss

        """
        self.mean_loss = loss.sum() / mask.sum()
        return self.mean_loss

    def set_up(self, model: torch.nn.Module, config: Union[CommonArgs, Munch]):
        r"""
        Training setup of optimizer and scheduler

        Args:
            model (torch.nn.Module): model for setup
            config (Union[CommonArgs, Munch]): munchified dictionary of args (:obj:`config.train.lr`, :obj:`config.metric`, :obj:`config.train.mile_stones`)

        Returns:
            None

        """
        self.model: torch.nn.Module = model
        # PORTS.md (OCC): optimise trainable params only -- a no-op unless a
        # mitigation froze part of the model (all params require grad upstream).
        self.optimizer = torch.optim.Adam(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=config.train.lr,
            weight_decay=config.train.weight_decay
        )
        # self.optimizer = torch.optim.Adam(
        #         [
        #             {'params': model.classifierS.parameters(), "lr": config.train.lr / 5, 'weight_decay': config.train.weight_decay},
        #             {'params': model.gnn_clf.parameters(), "lr": config.train.lr / 5, 'weight_decay': config.train.weight_decay},
        #             {'params': [p for name, p in model.named_parameters() if ('gnn_clf' not in name) and (('classifierS' not in name))]}
        #         ],
        #         lr=config.train.lr,
        #         weight_decay=config.train.weight_decay
        # )
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=config.train.mile_stones, gamma=0.1)

    def backward(self, loss):
        r"""
        Gradient backward process and parameter update.

        Args:
            loss: target loss
        """
        loss.backward()
        self.optimizer.step()


# ============================================================================== S3
def drive_norm_weights(loss, targets, mask, config):
    """Per-graph weights w (mean 1, detached) for the DATA TERM of the loss.

    MOTIVATION.  With a single logit t and CE, dCE/dt = -eps_G q_G with q_G the graph's
    error probability, so the drive reaching the extractor's logit ell_v is

        dL / d ell_v  =  - eps_G q_G a_v sigma'(ell_v),      a_v = dt / d s_v .

    The per-graph factor q_G means the class the classifier stays wrong about longer
    contributes more gradient.  If that is what produces the seed-stable class asymmetry
    in the alpha/beta decomposition, then cancelling q_G must make alpha class-symmetric.

    q is recovered from the loss itself -- no logits needed, so this works for BCE and
    softmax CE alike:   CE = -log p_correct   =>   q = 1 - exp(-CE).

    Returns None for mode "none" (caller leaves the loss untouched, bit-identical).
    """
    mode = str(getattr(config, "drive_norm", "none") or "none").lower()
    if mode == "none" or config.model.model_level == 'node':
        return None
    with torch.no_grad():
        B = loss.shape[0]
        ce = loss.detach().reshape(B, -1).sum(dim=1)
        if mask is not None:
            valid = mask.detach().reshape(B, -1).any(dim=1)
        else:
            valid = torch.ones(B, dtype=torch.bool, device=loss.device)
        qfloor = float(getattr(config, "drive_norm_qfloor", 1e-3))
        q = (1.0 - torch.exp(-ce)).clamp(min=qfloor, max=1.0)

        w = torch.ones_like(ce)
        if mode in ("invq", "invq_classbal", "shuffle"):
            w = 1.0 / q
        if mode in ("classbal", "invq_classbal"):
            cls = targets.detach().reshape(B, -1)[:, 0].long()
            uc = cls[valid].unique()
            if uc.numel() > 0:
                share = valid.sum().float() / uc.numel()
                for c in uc:
                    k = (cls == c) & valid
                    tot = w[k].sum()
                    if float(tot) > 0:
                        w = torch.where(k, w / tot * share, w)
        if mode == "shuffle":
            # PLACEBO: identical weight multiset, destroyed correspondence to q.
            idx = torch.randperm(B, device=w.device)
            w = w[idx]
        w = torch.where(valid, w, torch.zeros_like(w))
        m = w[valid].mean() if bool(valid.any()) else torch.ones((), device=w.device)
        w = w / (m + 1e-12)
    return w.reshape(B, *([1] * (loss.dim() - 1)))

