
from __future__ import annotations
 
import torch
from torch import distributed as dist
 
from llmcompressor.observers.base import Observer
from llmcompressor.observers.helpers import lerp
 
 
@Observer.register("luban_ema_absmax")
class LubanEMAAbsMaxObserver(Observer):
    _act_sync_dict = {
        "min_vals": dist.ReduceOp.AVG,
        "max_vals": dist.ReduceOp.AVG,
    }
 
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
 
        if not self.args.symmetric:
            raise ValueError(
                "luban_ema_absmax only supports symmetric quantization"
            )
 
        self.averaging_constant = float(
            self.args.observer_kwargs.get(
                "averaging_constant",
                0.01,
            )
        )
 
        if not 0.0 < self.averaging_constant <= 1.0:
            raise ValueError(
                "averaging_constant must be in (0, 1]"
            )
 
    def update_statistics_from_observed(
        self,
        observed: torch.Tensor,
    ) -> None:
        current_absmax = torch.amax(
            torch.abs(observed),
            dim=(0, -1),
        )
 
        if hasattr(self, "max_vals"):
            absmax = lerp(
                self.max_vals,
                current_absmax,
                self.averaging_constant,
            )
        else:
            absmax = current_absmax
 
        self.min_vals = -absmax
        self.max_vals = absmax