import torch


class GaussianDiffusion:
    def __init__(
        self,
        num_diffusion_timesteps=1000,
        schedule: str = "linear",
        device="cpu",
        dtype=torch.float32,
    ):
        self.num_diffusion_timesteps = num_diffusion_timesteps
        self.schedule = schedule
        self.device = device
        self.dtype = dtype
        self.initialize()

    def initialize(self):
        # BETAs & ALPHAs required at different places in the Algorithm.
        self.beta = self.get_betas()
        self.alpha = 1 - self.beta

        self.sqrt_beta = torch.sqrt(self.beta)
        self.alpha_cumulative = torch.cumprod(self.alpha, dim=0)
        self.sqrt_alpha_cumulative = torch.sqrt(self.alpha_cumulative)
        self.one_by_sqrt_alpha = 1.0 / torch.sqrt(self.alpha)
        self.sqrt_one_minus_alpha_cumulative = torch.sqrt(1 - self.alpha_cumulative)

    def get_betas(self):
        """Creates betas given th specified noise schedule."""
        if self.schedule == "linear":
            # Linear schedule from Ho et al, extended to work for any number of diffusion steps.
            scale = 1000 / self.num_diffusion_timesteps
            beta_start = scale * 1e-4
            beta_end = scale * 0.02
            return torch.linspace(
                start=beta_start,
                end=beta_end,
                steps=self.num_diffusion_timesteps,
                dtype=self.dtype,
                device=self.device,
            )

        elif self.schedule == "cosine":
            # Cosine schedule from improved diffusion
            # https://github.com/openai/improved-diffusion/blob/main/improved_diffusion/gaussian_diffusion.py
            max_beta = torch.tensor(0.999)
            alpha_bar = lambda t: torch.cos((t + 0.008) / 1.008 * torch.pi / 2) ** 2

            ts = torch.arange(
                0, self.num_diffusion_timesteps, device=self.device, dtype=self.dtype
            )
            t1 = ts / self.num_diffusion_timesteps
            t2 = (ts + 1) / self.num_diffusion_timesteps
            betas = torch.minimum(1 - alpha_bar(t2) / alpha_bar(t1), max_beta)
            return betas

        else:
            raise NotImplementedError(f"Unknown noise schedule: {self.schedule}")

    def forward_diffusion(self, x0: torch.Tensor, timesteps: torch.Tensor):
        # some shape magic to deal with one or vector of timesteps

        x0 = x0.to(self.device)
        timesteps = timesteps.to(self.device)

        B, _, *rest = x0.shape
        view_shape = (B, 1, *[1 for _ in rest])
        ts = timesteps.to(dtype=torch.long)
        if len(timesteps.shape) == 0:
            ts = ts.tile(B)

        eps = torch.randn_like(x0)  # Noise

        mean = self.sqrt_alpha_cumulative[ts].view(view_shape).expand_as(x0) * x0
        std_dev = (
            self.sqrt_one_minus_alpha_cumulative[ts].view(view_shape).expand_as(x0)
        )
        sample = mean + std_dev * eps  # scaled inputs * scaled noise
        return sample, eps
