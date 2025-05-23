import torch
import math


class DiffusionSampler:
    def __init__(self, model, num_steps=1000, beta_schedule="cosine", device="cuda"):
        super().__init__()
        self.model = model
        self.num_steps = num_steps
        self.beta_schedule = beta_schedule
        self.device = device

        # 调用 _compute_betas 获取 betas 并计算 alpha_bars
        self.betas = self._compute_betas().to(device)
        self.alpha_bars = torch.cumprod(1. - self.betas, dim=0).to(device)

    def _compute_betas(self):
        """根据 beta_schedule 计算 betas"""
        if self.beta_schedule == "linear":
            scale = 1000.0 / self.num_steps
            beta_start = scale * 1e-4
            beta_end = scale * 2e-2
            betas = torch.linspace(beta_start, beta_end, self.num_steps)
        elif self.beta_schedule == "cosine":
            steps = self.num_steps
            t = torch.arange(steps + 1, dtype=torch.float32)
            alphas_cumprod = torch.cos((t / steps + 0.008) / (1 + 0.008) * math.pi / 2) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            betas = torch.clip(betas, 0, 0.999)
        else:
            raise ValueError(f"Unsupported beta schedule: {self.beta_schedule}")
        return betas
    
    def sample(self, x_shape, cond=None, text=None, steps=50, eta=0.0, cfg_strength=2.0, sway_sampling_coef=-1):
        """
        DDIM Sampler
        Args:
            x_shape: shape of input mel [B, T, D]
            cond: condition audio embedding
            text: tokenized text input
            steps: number of sampling steps
            eta: controls variance in DDIM (eta=0 is deterministic)
            cfg_strength: classifier-free guidance scale
            sway_sampling_coef: sway sampling coefficient for DiT
        Returns:
            denoised mel spectrogram
        """
        device = next(self.model.parameters()).device
        batch_size = x_shape[0]
        x = torch.randn(x_shape, device=device)

        # Time step list
        times = torch.linspace(self.num_steps - 1, 0, steps).long().to(device)

        for i, t in enumerate(times):
            with torch.no_grad():
                # Classifier-Free Guidance
                if cfg_strength > 0:
                    uncond_logits = self.model(
                        x=x,
                        cond=cond,
                        text=torch.zeros_like(text),
                        time=t.unsqueeze(0).expand(batch_size),
                    )
                    cond_logits = self.model(
                        x=x,
                        cond=cond,
                        text=text,
                        time=t.unsqueeze(0).expand(batch_size),
                    )
                    predicted_noise = uncond_logits + cfg_strength * (cond_logits - uncond_logits)
                else:
                    predicted_noise = self.model(
                        x=x,
                        cond=cond,
                        text=text,
                        time=t.unsqueeze(0).expand(batch_size),
                    )

                # DDIM inversion
                alpha_bar_t = self.alpha_bars[t].view(-1, 1, 1)
                alpha_bar_prev = self.alpha_bars[
                    times[i + 1] if i + 1 < len(times) else 0
                ].view(-1, 1, 1)

                sigma = (
                    eta
                    * ((1 - alpha_bar_prev) / (1 - alpha_bar_t))
                    * (1 - alpha_bar_t / alpha_bar_prev)
                ).sqrt()

                pred_x0 = (x - (1 - alpha_bar_t).sqrt() * predicted_noise) / alpha_bar_t.sqrt()
                dir_xt = (1 - alpha_bar_prev - sigma ** 2).sqrt() * predicted_noise
                noise = torch.randn_like(x) if i < len(times) - 1 and eta > 0 else 0

                x = alpha_bar_prev.sqrt() * pred_x0 + dir_xt + sigma * noise

        return x, None
    
class DiTSamplerWrapper(torch.nn.Module):
    def __init__(self, model, num_steps=1000, beta_schedule="cosine", device="cuda"):
        super().__init__()
        self.model = model.to(device)
        self.diffusion_sampler = DiffusionSampler(
            self.model, num_steps, beta_schedule, device=device  # 新增参数
        )
        self.device = device

    def sample(self, cond, text, duration, steps=32, cfg_strength=2.0, sway_sampling_coef=-1, **kwargs):
        """
        包装后的 sample 方法，兼容 infer_process 流程
        """
        x_shape = (1, duration, 100)  # mel dim = 100
        with torch.inference_mode():
            mel, _ = self.diffusion_sampler.sample(
                x_shape=x_shape,
                cond=cond,
                text=text,
                steps=steps,
                cfg_strength=cfg_strength,
                sway_sampling_coef=sway_sampling_coef,
            )
        return mel, None