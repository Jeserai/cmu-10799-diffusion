"""
Flow Matching implementation (continuous-time generative model).
"""

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseMethod


class FlowMatching(BaseMethod):
    """
    Simple flow matching with linear interpolation between data and noise.

    We sample a time t ~ Uniform(0, 1), form:
        x_t = (1 - t) * x_0 + t * x_1,  x_1 ~ N(0, I)
    and regress the velocity:
        v_t = x_1 - x_0
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        time_scale: float = 1000.0,
    ):
        super().__init__(model, device)
        self.time_scale = float(time_scale)

    def compute_loss(
        self, x_0: torch.Tensor, **kwargs
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        batch_size = x_0.shape[0]
        t = torch.rand(batch_size, device=x_0.device)
        t_broadcast = t.view(batch_size, *([1] * (x_0.ndim - 1)))

        x_1 = torch.randn_like(x_0)
        x_t = (1.0 - t_broadcast) * x_0 + t_broadcast * x_1
        v_target = x_1 - x_0

        v_pred = self.model(x_t, t * self.time_scale)
        loss = F.mse_loss(v_pred, v_target)
        metrics = {"loss": loss.detach()}
        return loss, metrics

    @torch.no_grad()
    def sample(
        self,
        batch_size: int,
        image_shape: Tuple[int, int, int],
        num_steps: int = 1000,
        **kwargs,
    ) -> torch.Tensor:
        self.eval_mode()

        x = torch.randn((batch_size, *image_shape), device=self.device)
        dt = -1.0 / float(num_steps)
        for i in range(num_steps):
            t = 1.0 - i / float(num_steps)
            t_batch = torch.full(
                (batch_size,), t, device=self.device, dtype=torch.float32
            )
            v = self.model(x, t_batch * self.time_scale)
            x = x + v * dt

        return x

    def sample_guided(
        self,
        batch_size: int,
        image_shape: Tuple[int, int, int],
        classifier: nn.Module,
        target_class_idx: int,
        guidance_scale: float = 1.0,
        num_steps: int = 1000,
        guidance_mode: str = "fmps",
    ) -> torch.Tensor:
        """
        Classifier-guided sampling with Euler integration.
        """
        self.eval_mode()
        classifier.eval()

        x = torch.randn((batch_size, *image_shape), device=self.device)
        dt = -1.0 / float(num_steps)

        for i in range(num_steps):
            t = 1.0 - i / float(num_steps)
            t_batch = torch.full(
                (batch_size,), t, device=self.device, dtype=torch.float32
            )

            x = x.detach().requires_grad_(True)
            with torch.no_grad():
                v_base = self.model(x, t_batch * self.time_scale)
            logits = classifier(x, t_batch)
            if guidance_mode == "logit":
                score = logits[:, target_class_idx]
                grad = torch.autograd.grad(score.sum(), x, create_graph=False)[0]
                v = v_base + guidance_scale * grad
            else:
                log_prob = F.logsigmoid(logits[:, target_class_idx])
                grad = torch.autograd.grad(log_prob.sum(), x, create_graph=False)[0]
                if guidance_mode == "fmps":
                    # FMPS-style correction: v_guided = v_base - ((1 - t) / t) * grad log p(y|x)
                    t_safe = torch.tensor(t, device=self.device).clamp(min=1e-5)
                    correction = (1.0 - t_safe) / t_safe
                    v = v_base - guidance_scale * correction * grad
                else:
                    v = v_base + guidance_scale * grad
            x = x + v * dt

        return x.detach()

    def state_dict(self) -> Dict:
        state = super().state_dict()
        state["time_scale"] = self.time_scale
        return state

    @classmethod
    def from_config(
        cls, model: nn.Module, config: dict, device: torch.device
    ) -> "FlowMatching":
        fm_config = config.get("flow_matching", config)
        return cls(
            model=model,
            device=device,
            time_scale=fm_config.get("time_scale", 1000.0),
        )

