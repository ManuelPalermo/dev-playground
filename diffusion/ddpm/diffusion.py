import torch


class GaussianDiffusion:
    """Applies gaussian diffusion to samples."""

    def __init__(
        self,
        num_diffusion_timesteps=1000,
        schedule: str = "linear",
        beta_1: float = 1e-4,
        beta_T: float = 0.02,
        device="cpu",
        dtype=torch.float32,
    ):
        """Constructor."""
        self.num_diffusion_timesteps = num_diffusion_timesteps
        self.schedule = schedule
        self.beta_1 = beta_1
        self.beta_T = beta_T
        self.device = device
        self.dtype = dtype
        self._initialize()

    def _initialize(self):
        """Create BETAs & ALPHAs required at different places in the Algorithm."""
        self.beta = self._get_betas()
        self.alpha = 1 - self.beta

        self.sqrt_beta = torch.sqrt(self.beta)
        self.alpha_cumulative = torch.cumprod(self.alpha, dim=0)
        self.sqrt_alpha_cumulative = torch.sqrt(self.alpha_cumulative)
        self.one_by_sqrt_alpha = 1.0 / torch.sqrt(self.alpha)
        self.sqrt_one_minus_alpha_cumulative = torch.sqrt(1 - self.alpha_cumulative)

    def _get_betas(self):
        """Creates betas given th specified noise schedule."""
        if self.schedule == "linear":
            # Linear schedule from Ho et al, extended to work for any number of diffusion steps.
            scale = 1000 / self.num_diffusion_timesteps
            beta_start = scale * self.beta_1
            beta_end = scale * self.beta_T
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
            max_beta = torch.tensor(1 - self.beta_T)
            alpha_bar = lambda t: torch.cos((t + self.beta_1) / (1 + self.beta_1) * torch.pi / 2) ** 2

            ts = torch.arange(0, self.num_diffusion_timesteps, device=self.device, dtype=self.dtype)
            t1 = ts / self.num_diffusion_timesteps
            t2 = (ts + 1) / self.num_diffusion_timesteps
            betas = torch.minimum(1 - alpha_bar(t2) / alpha_bar(t1), max_beta)
            return betas

        else:
            raise NotImplementedError(f"Unknown noise schedule: {self.schedule}")

    def forward_diffusion(self, x0: torch.Tensor, timesteps: torch.Tensor):
        """Performs forward diffusion, returning corrupted samples and noise applied at a given timestep.

        Args:
            x0: initial images.
            timesteps: Either a single timestep of vector of timesteps for each image.
        """
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
        std_dev = self.sqrt_one_minus_alpha_cumulative[ts].view(view_shape).expand_as(x0)
        sample = mean + std_dev * eps  # scaled inputs * scaled noise
        return sample, eps
