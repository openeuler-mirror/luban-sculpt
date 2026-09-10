from __future__ import annotations
 
import torch
from torch import distributed as dist
 
from llmcompressor.observers.base import Observer
from llmcompressor.observers.helpers import lerp
 
 
@Observer.register("luban_percentile")
class LubanPercentileObserver(Observer):
    """
    百分位裁剪 Observer。
 
    percentile=0.999 表示使用：
        0.1% 分位点作为下界
        99.9% 分位点作为上界
    """
 
    _act_sync_dict = {
        "min_vals": dist.ReduceOp.AVG,
        "max_vals": dist.ReduceOp.AVG,
    }
 
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
 
        self.percentile = float(
            self.args.observer_kwargs.get(
                "percentile",
                0.999,
            )
        )
 
        self.averaging_constant = float(
            self.args.observer_kwargs.get(
                "averaging_constant",
                0.01,
            )
        )
 
        if not 0.5 < self.percentile <= 1.0:
            raise ValueError(
                "percentile must be in (0.5, 1.0]"
            )
 
        if not 0.0 < self.averaging_constant <= 1.0:
            raise ValueError(
                "averaging_constant must be in (0, 1]"
            )
 
    def update_statistics_from_observed(
        self,
        observed: torch.Tensor,
    ) -> None:
        """
        observed 的形状：
 
            (num_observations, *qparam_shape, group_size)
 
        把中间 qparam 维度保留，将第0维和最后一维合并，
        然后计算分位点。
        """
 
        qparam_shape = tuple(observed.shape[1:-1])
 
        permutation = (
            *range(1, observed.ndim - 1),
            0,
            observed.ndim - 1,
        )
 
        grouped = (
            observed
            .permute(permutation)
            .reshape(qparam_shape + (-1,))
            .float()
        )
 
        lower_q = 1.0 - self.percentile
        upper_q = self.percentile
 
        current_min = torch.quantile(
            grouped,
            lower_q,
            dim=-1,
        )
        current_max = torch.quantile(
            grouped,
            upper_q,
            dim=-1,
        )
 
        if hasattr(self, "min_vals"):
            min_vals = lerp(
                self.min_vals,
                current_min,
                self.averaging_constant,
            )
            max_vals = lerp(
                self.max_vals,
                current_max,
                self.averaging_constant,
            )
        else:
            min_vals = current_min
            max_vals = current_max
 
        self.min_vals = min_vals
        self.max_vals = max_vals