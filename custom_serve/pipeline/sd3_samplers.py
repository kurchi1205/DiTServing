import sched
import torch
import torch.nn as nn
import math
import matplotlib.pyplot as plt


class KarrasScheduler(nn.Module):
    def __init__(self, sigma_min=0.002, sigma_max=1.0, rho=7.0, num_steps=1000):
        """
        Initialize the Karras noise scheduler.

        Args:
            sigma_min (float): Minimum noise level.
            sigma_max (float): Maximum noise level.
            rho (float): Controls the shape of the curve.
            num_steps (int): Number of inference steps.
        """
        super().__init__()
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho = rho
        self.num_steps = num_steps

        # Create the Karras sigma schedule
        self.register_buffer("sigmas", self._create_schedule())

    def _create_schedule(self):
        """
        Generate the Karras noise schedule using the formula:
        sigma(t) = [sigma_max^(1/rho) + t * (sigma_min^(1/rho) - sigma_max^(1/rho))]^rho
        """
        t = torch.linspace(0, 1, self.num_steps)
        sigma_schedule = (
            (self.sigma_max ** (1 / self.rho)) +
            (1 - t) * (self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho))
        ) ** self.rho
        return sigma_schedule

    def get_sigma(self, step):
        """
        Get sigma for a given step index.
        """
        return self.sigmas[step]

    def get_schedule(self):
        """
        Get the full schedule as a tensor.
        """
        return self.sigmas.clone()

    def calculate_denoised(self, sigma, model_output, model_input):
        """
        Basic denoising rule: x0 = x - sigma * eps
        """
        sigma = sigma.view(sigma.shape[:1] + (1,) * (model_output.ndim - 1))
        return model_input - model_output * sigma

    def noise_scaling(self, sigma, noise, latent_image):
        """
        Combine latent and noise as: x = sigma * noise + (1 - sigma) * latent
        """
        return sigma * noise + (1.0 - sigma) * latent_image


class ModelSamplingDiscreteFlow(torch.nn.Module):
    """Helper for sampler scheduling (ie timestep/sigma calculations) for Discrete Flow models"""

    def __init__(self, shift=1.0):
        super().__init__()
        self.shift = shift
        timesteps = 1000
        ts = self.sigma(torch.arange(1, timesteps + 1, 1))
        self.register_buffer("sigmas", ts)

    @property
    def sigma_min(self):
        return self.sigmas[0]

    @property
    def sigma_max(self):
        return self.sigmas[-1]
    

    def get_schedule(self):
        return self.sigmas.clone()

    def timestep(self, sigma):
        return sigma * 1000

    def sigma(self, timestep: torch.Tensor):
        timestep = timestep / 1000.0
        if self.shift == 1.0:
            return timestep
        return (self.shift * (timestep ** 1.2))/ (1 + (self.shift  - 1) * (timestep ** 1.2))

    def calculate_denoised(self, sigma, model_output, model_input):
        sigma = sigma.view(sigma.shape[:1] + (1,) * (model_output.ndim - 1))
        return model_input - model_output * sigma

    def noise_scaling(self, sigma, noise, latent_image, max_denoise=False):
        return sigma * noise + (1.0 - sigma) * latent_image
    

class ExponentialScheduler(nn.Module):
    def __init__(self, sigma_min=0.002, sigma_max=1.0, num_steps=1000):
        super().__init__()
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.num_steps = num_steps

        # Build exponential schedule
        self.register_buffer("sigmas", self._create_schedule())

    def _create_schedule(self):
        t = torch.linspace(0, 1, self.num_steps)
        return self.sigma_max * (self.sigma_min / self.sigma_max) ** (1 - t)
    
    def get_schedule(self):
        return self.sigmas.clone()

    def sigma(self, timestep: torch.Tensor):
        # Accepts raw step index [0, T], normalizes internally
        t = timestep.float() / self.num_steps
        return self.sigma_max * (self.sigma_min / self.sigma_max) ** t

    def calculate_denoised(self, sigma, model_output, model_input):
        sigma = sigma.view(sigma.shape[:1] + (1,) * (model_output.ndim - 1))
        return model_input - model_output * sigma

    def noise_scaling(self, sigma, noise, latent_image):
        return sigma * noise + (1.0 - sigma) * latent_image


class WarmCosineScheduler(nn.Module):
    def __init__(self, sigma_min=0.002, sigma_max=1.0, num_steps=1000, warmup_steps=50):
        super().__init__()
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.num_steps = num_steps
        self.warmup_steps = warmup_steps

        self.register_buffer("sigmas", self._create_schedule())

    def _create_schedule(self):
        sigmas = torch.zeros(self.num_steps, dtype=torch.float32)

        # Cosine decay: from σ_max to σ_min
        decay_steps = self.num_steps - self.warmup_steps
        t = torch.linspace(0, math.pi / 2, decay_steps)
        cosine_decay = self.sigma_min + (self.sigma_max - self.sigma_min) * (torch.cos(t) ** 2)

        # Fill schedule in reverse: end of denoising first (low t = low sigma)
        sigmas[:decay_steps] = cosine_decay.flip(0)

        # Warm-up at the beginning (high t = high sigma)
        sigmas[decay_steps:] = self.sigma_max

        return sigmas


    def get_schedule(self):
        return self.sigmas.clone()

    def sigma(self, timestep):
        return self.sigmas[timestep]

    def calculate_denoised(self, sigma, model_output, model_input):
        sigma = sigma.view(sigma.shape[:1] + (1,) * (model_output.ndim - 1))
        return model_input - model_output * sigma

    def noise_scaling(self, sigma, noise, latent_image):
        return sigma * noise + (1.0 - sigma) * latent_image
    

if __name__ == "__main__":
    scheduler_karras = KarrasScheduler(num_steps=1000, rho=2)
    scheduler_exp = ExponentialScheduler(num_steps=1000)
    scheduler_warmcosine = WarmCosineScheduler(num_steps=1000, warmup_steps=200)
    scheduler = ModelSamplingDiscreteFlow(shift=5)

    sigmas_karras = scheduler_karras.get_schedule()
    sigmas_exp = scheduler_exp.get_schedule()
    sigmas_warmcosine = scheduler_warmcosine.get_schedule()
    sigmas = scheduler.get_schedule()

    plt.plot(sigmas_karras.cpu().numpy(), label='Karras Scheduler')
    plt.plot(sigmas_exp.cpu().numpy(), label='Exponential Scheduler')
    plt.plot(sigmas_warmcosine.cpu().numpy(), label='Warm Cosine Scheduler')
    plt.plot(sigmas.cpu().numpy(), label='Discrete Flow Scheduler')

    # Add labels and legend
    plt.title("Noise Schedules Comparison")
    plt.xlabel("Timestep")
    plt.ylabel("Sigma (Noise Level)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("noise_schedule_comparison.png")