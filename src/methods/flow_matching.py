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
        target_class_idx: int | list[int],
        secondary_target_class_idx: int | list[int] | None = None,
        guidance_scale: float = 1.0,
        secondary_guidance_scale: float = 0.0,
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

        def project_onto(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
            reduce_dims = tuple(range(1, a.ndim))
            dot_ab = (a * b).sum(dim=reduce_dims, keepdim=True)
            dot_bb = (b * b).sum(dim=reduce_dims, keepdim=True)
            return (dot_ab / (dot_bb + eps)) * b

        def normalize_indices(indices: int | list[int]) -> list[int]:
            return indices if isinstance(indices, list) else [indices]

        for i in range(num_steps):
            t = 1.0 - i / float(num_steps)
            t_batch = torch.full(
                (batch_size,), t, device=self.device, dtype=torch.float32
            )

            x = x.detach().requires_grad_(True)
            with torch.no_grad():
                v_base = self.model(x, t_batch * self.time_scale)
            logits = classifier(x, t_batch)
            target_indices = normalize_indices(target_class_idx)

            if guidance_mode == "orthogonal":
                log_prob_1 = F.logsigmoid(logits[:, target_indices]).sum(dim=1)
                g1 = torch.autograd.grad(log_prob_1.sum(), x, create_graph=False)[0]
                g1_orth = g1 - project_onto(g1, v_base)

                if secondary_target_class_idx is not None and secondary_guidance_scale != 0.0:
                    secondary_indices = normalize_indices(secondary_target_class_idx)
                    log_prob_2 = F.logsigmoid(logits[:, secondary_indices]).sum(dim=1)
                    g2 = torch.autograd.grad(log_prob_2.sum(), x, create_graph=False)[0]
                    g2_orth = g2 - project_onto(g2, v_base) - project_onto(g2, g1_orth)
                    v = v_base + guidance_scale * g1_orth + secondary_guidance_scale * g2_orth
                else:
                    v = v_base + guidance_scale * g1_orth
            elif guidance_mode == "logit":
                score = logits[:, target_indices].sum(dim=1)
                grad = torch.autograd.grad(score.sum(), x, create_graph=False)[0]
                v = v_base + guidance_scale * grad
            elif guidance_mode == "parallel":
                # Project the classifier gradient onto the base velocity and discard orthogonal noise
                log_prob_1 = F.logsigmoid(logits[:, target_indices]).sum(dim=1)
                g1 = torch.autograd.grad(log_prob_1.sum(), x, create_graph=False)[0]
                g1_par = project_onto(g1, v_base)

                if secondary_target_class_idx is not None and secondary_guidance_scale != 0.0:
                    secondary_indices = normalize_indices(secondary_target_class_idx)
                    log_prob_2 = F.logsigmoid(logits[:, secondary_indices]).sum(dim=1)
                    g2 = torch.autograd.grad(log_prob_2.sum(), x, create_graph=False)[0]
                    g2_par = project_onto(g2, v_base)
                    v = v_base + guidance_scale * g1_par + secondary_guidance_scale * g2_par
                else:
                    v = v_base + guidance_scale * g1_par
            elif guidance_mode == "pcgrad":
                # Projecting Conflicting Gradients (PCGrad) for two-attribute guidance
                # Compute surrogate-corrected log-probs and gradients for both targets
                if secondary_target_class_idx is None:
                    # Fallback to single-attribute grad if no secondary provided
                    log_prob = F.logsigmoid(logits[:, target_indices]).sum(dim=1)
                    g = torch.autograd.grad(log_prob.sum(), x, create_graph=False)[0]
                    v = v_base + guidance_scale * g
                else:
                    # compute gradients
                    target_indices_1 = normalize_indices(target_class_idx)
                    target_indices_2 = normalize_indices(secondary_target_class_idx)
                    log_prob_1 = F.logsigmoid(logits[:, target_indices_1]).sum(dim=1)
                    log_prob_2 = F.logsigmoid(logits[:, target_indices_2]).sum(dim=1)
                    g1 = torch.autograd.grad(log_prob_1.sum(), x, create_graph=False)[0]
                    g2 = torch.autograd.grad(log_prob_2.sum(), x, create_graph=False)[0]

                    # vectorized dot products and norms per-sample
                    reduce_dims = tuple(range(1, g1.ndim))
                    dot12 = (g1 * g2).sum(dim=reduce_dims, keepdim=True)
                    g1_sq = (g1 * g1).sum(dim=reduce_dims, keepdim=True)
                    g2_sq = (g2 * g2).sum(dim=reduce_dims, keepdim=True)

                    # conflict mask where dot < 0
                    conflict = (dot12 < 0.0).to(x.dtype)

                    eps = 1e-8
                    # compute projection terms (safe with broadcasting)
                    proj_factor_1 = dot12 / (g2_sq + eps)
                    proj_factor_2 = dot12 / (g1_sq + eps)

                    # apply PCGrad only where conflict is true
                    g1_star = g1 - (proj_factor_1 * g2) * conflict
                    g2_star = g2 - (proj_factor_2 * g1) * conflict

                    v = v_base + guidance_scale * g1_star + secondary_guidance_scale * g2_star
            elif guidance_mode == "rescaling":
                # Rescaling guidance: compose gradients and match statistical footprint of single gradient
                if secondary_target_class_idx is None:
                    log_prob = F.logsigmoid(logits[:, target_indices]).sum(dim=1)
                    g = torch.autograd.grad(log_prob.sum(), x, create_graph=False)[0]
                    v = v_base + guidance_scale * g
                else:
                    # compute gradients for both targets
                    target_indices_1 = normalize_indices(target_class_idx)
                    target_indices_2 = normalize_indices(secondary_target_class_idx)
                    log_prob_1 = F.logsigmoid(logits[:, target_indices_1]).sum(dim=1)
                    log_prob_2 = F.logsigmoid(logits[:, target_indices_2]).sum(dim=1)
                    g1 = torch.autograd.grad(log_prob_1.sum(), x, create_graph=False)[0]
                    g2 = torch.autograd.grad(log_prob_2.sum(), x, create_graph=False)[0]

                    # compose gradients
                    g_sum = g1 + g2

                    # compute std per sample across C, H, W dims (keep batch dim for rescaling)
                    reduce_dims = tuple(range(1, g1.ndim))
                    std_g1 = g1.std(dim=reduce_dims, keepdim=True)
                    std_g_sum = g_sum.std(dim=reduce_dims, keepdim=True)

                    eps = 1e-8
                    # rescale: g_sum * (std_g1 / std_g_sum)
                    g_rescaled = g_sum * (std_g1 / (std_g_sum + eps))

                    v = v_base + guidance_scale * g_rescaled
            elif guidance_mode == "sequential":
                # Sequential split-step guidance: split each Euler step in half
                # Sub-step A: apply first condition, then re-evaluate for sub-step B
                if secondary_target_class_idx is None:
                    # Fallback to single-attribute if no secondary provided
                    log_prob = F.logsigmoid(logits[:, target_indices]).sum(dim=1)
                    g = torch.autograd.grad(log_prob.sum(), x, create_graph=False)[0]
                    t_safe = torch.tensor(t, device=self.device).clamp(min=1e-5)
                    correction = (1.0 - t_safe) / t_safe
                    v = v_base - guidance_scale * correction * g
                    x = x + v * dt
                else:
                    # Sub-step A: apply first condition over dt/2
                    target_indices_1 = normalize_indices(target_class_idx)
                    log_prob_1 = F.logsigmoid(logits[:, target_indices_1]).sum(dim=1)
                    g1 = torch.autograd.grad(log_prob_1.sum(), x, create_graph=False)[0]
                    t_safe = torch.tensor(t, device=self.device).clamp(min=1e-5)
                    correction_a = (1.0 - t_safe) / t_safe
                    v_a = v_base - guidance_scale * correction_a * g1
                    x_mid = x + v_a * (dt / 2.0)

                    # Re-evaluate at midpoint for sub-step B
                    t_mid = t + (dt / 2.0)  # time advances by dt/2
                    t_mid_batch = torch.full(
                        (batch_size,), t_mid, device=self.device, dtype=torch.float32
                    )
                    x_mid = x_mid.detach().requires_grad_(True)
                    with torch.no_grad():
                        v_base_b = self.model(x_mid, t_mid_batch * self.time_scale)
                    logits_b = classifier(x_mid, t_mid_batch)

                    # Sub-step B: apply second condition over remaining dt/2
                    target_indices_2 = normalize_indices(secondary_target_class_idx)
                    log_prob_2 = F.logsigmoid(logits_b[:, target_indices_2]).sum(dim=1)
                    g2 = torch.autograd.grad(log_prob_2.sum(), x_mid, create_graph=False)[0]
                    t_mid_safe = torch.tensor(t_mid, device=self.device).clamp(min=1e-5)
                    correction_b = (1.0 - t_mid_safe) / t_mid_safe
                    v_b = v_base_b - secondary_guidance_scale * correction_b * g2
                    x = x_mid + v_b * (dt / 2.0)
                    # Skip the normal x update at the end since we've already done both sub-steps
                    continue
            elif guidance_mode == "manifold":
                # Manifold projection guidance: estimate x0 from guided step, then re-noise to current t
                # Compute guided velocity (single or two-attr)
                if secondary_target_class_idx is None:
                    log_prob = F.logsigmoid(logits[:, target_indices]).sum(dim=1)
                    g = torch.autograd.grad(log_prob.sum(), x, create_graph=False)[0]
                    v_guided = v_base + guidance_scale * g
                else:
                    target_indices_1 = normalize_indices(target_class_idx)
                    target_indices_2 = normalize_indices(secondary_target_class_idx)
                    log_prob_1 = F.logsigmoid(logits[:, target_indices_1]).sum(dim=1)
                    log_prob_2 = F.logsigmoid(logits[:, target_indices_2]).sum(dim=1)
                    g1 = torch.autograd.grad(log_prob_1.sum(), x, create_graph=False)[0]
                    g2 = torch.autograd.grad(log_prob_2.sum(), x, create_graph=False)[0]
                    v_guided = v_base + guidance_scale * g1 + secondary_guidance_scale * g2

                # Take the guided Euler step to get x_guided
                x_guided = x + v_guided * dt

                # Estimate projected clean image x0_hat from model at x_guided
                with torch.no_grad():
                    v_proj = self.model(x_guided, t_batch * self.time_scale)
                    # flow-matching reparam: x_t = (1 - t) * x0 + t * x1  -> x0_hat = (x_t - t*x1) / (1 - t)
                    # however user-provided formula: x0_hat = x_guided - t * v_proj
                    x0_hat = x_guided - t * v_proj

                # Re-noise to the next time level (t_next = t + dt)
                t_next = t + dt
                # sample fresh noise at the same device/shape
                noise = torch.randn_like(x0_hat, device=x0_hat.device)
                x = (1.0 - t_next) * x0_hat + t_next * noise
                # we've already advanced x for this step
                continue
            else:
                log_prob = F.logsigmoid(logits[:, target_indices]).sum(dim=1)
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

