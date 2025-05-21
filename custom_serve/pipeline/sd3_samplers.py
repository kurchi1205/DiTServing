import torch
import torch.nn as nn
import math
import matplotlib.pyplot as plt


class KarrasScheduler(nn.Module):
    """Helper for sampler scheduling using Karras-style sigma interpolation"""

    def __init__(self, sigma_min=0.002, sigma_max=1.0, rho=7.0, num_steps=1000):
        super().__init__()
        self.sigma_min_val = sigma_min
        self.sigma_max_val = sigma_max
        self.rho = rho
        self.num_steps = num_steps

        # Build the schedule
        sigmas = self._create_schedule()
        self.register_buffer("sigmas", sigmas)

    def _create_schedule(self):
        # Build reversed time: t = 1 to 0 to ensure t=0 → sigma_min, t=T-1 → sigma_max
        t = torch.linspace(1, 0, self.num_steps)
        sigma_schedule = (
            (self.sigma_max_val ** (1 / self.rho)) +
            t * (self.sigma_min_val ** (1 / self.rho) - self.sigma_max_val ** (1 / self.rho))
        ) ** self.rho
        return sigma_schedule

    @property
    def sigma_min(self):
        return self.sigmas[-1]

    @property
    def sigma_max(self):
        return self.sigmas[0]

    def get_schedule(self):
        return self.sigmas.clone()

    def timestep(self, sigma):
        """
        Convert a given sigma to a normalized timestep in [0, 1].
        t = 0 corresponds to sigma_min, t = 1 to sigma_max.
        """
        sigma_root = sigma ** (1 / self.rho)
        sigma_max_root = self.sigma_max_val ** (1 / self.rho)
        sigma_min_root = self.sigma_min_val ** (1 / self.rho)

        t = (sigma_root - sigma_min_root) / (sigma_max_root - sigma_min_root)
        return t * self.num_steps  # to match DiscreteFlow's scale

    def sigma(self, timestep: torch.Tensor):
        """
        Convert timestep ∈ [0, num_steps] → sigma using Karras formula.
        """
        t = 1 - timestep / self.num_steps  # reversed time
        return (
            (self.sigma_max_val ** (1 / self.rho)) +
            t * (self.sigma_min_val ** (1 / self.rho) - self.sigma_max_val ** (1 / self.rho))
        ) ** self.rho

    def calculate_denoised(self, sigma, model_output, model_input):
        sigma = sigma.view(sigma.shape[:1] + (1,) * (model_output.ndim - 1))
        return model_input - model_output * sigma

    def noise_scaling(self, sigma, noise, latent_image, max_denoise=False):
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
    

    def get_schedule(self, num_steps):
        req_sigmas = self.sigmas.clone()[:num_steps]
        return req_sigmas

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
    
    def get_schedule(self, num_steps):
        req_sigmas = self.sigmas.clone()[:num_steps]
        return req_sigmas

    def sigma(self, timestep: torch.Tensor):
        # Accepts raw step index [0, T], normalizes internally
        t = timestep.float() / self.num_steps
        return self.sigma_max * (self.sigma_min / self.sigma_max) ** t

    def calculate_denoised(self, sigma, model_output, model_input):
        sigma = sigma.view(sigma.shape[:1] + (1,) * (model_output.ndim - 1))
        return model_input - model_output * sigma

    def noise_scaling(self, sigma, noise, latent_image):
        return sigma * noise + (1.0 - sigma) * latent_image


class WarmupCosineScheduler(nn.Module):
    def __init__(self, shift=1.0, min_beta=0.002, max_beta=0.999, num_steps=1000, warmup_steps=50):
        super().__init__()
        self.shift = shift
        self.min_beta = min_beta   # Minimum noise level (at the beginning)
        self.max_beta = max_beta   # Maximum noise level (at the end and warmup)
        self.num_steps = num_steps
        self.warmup_steps = min(warmup_steps, num_steps)
        
        # Create the schedule with the correct orientation
        ts = self._create_schedule_with_warmup()
        self.register_buffer("sigmas", ts)
    
    def _create_schedule_with_warmup(self):
        """
        Create schedule with the correct orientation:
        - Starting with low sigma (timestep 0)
        - Ending with high sigma (timestep 1000)
        - With warmup section at the end
        """
        # Calculate main schedule (excluding warmup)
        main_steps = self.num_steps - self.warmup_steps
        
        if main_steps > 0:
            # Create main schedule with correct orientation
            # We use a flipped cosine curve: sigma increases from min_beta to max_beta
            sigmas = torch.zeros(self.num_steps, dtype=torch.float32)
            
            # Calculate main portion - from low to high sigma
            for i in range(main_steps):
                # Normalized t from 0 to 1, but we flip the cosine formula
                # to get low->high instead of high->low
                t = i / main_steps
                sigmas[i] = self._flipped_cosine_sigma(t)
            
            # Set warmup portion to the maximum sigma value
            if self.warmup_steps > 0:
                sigmas[main_steps:] = self.max_beta
                
            return sigmas
        else:
            # If all steps are warmup, use constant high sigma
            return torch.ones(self.num_steps) * self.max_beta
    
    def _flipped_cosine_sigma(self, t):
        """
        Calculate sigma using flipped cosine formula to get low->high orientation
        t: normalized timestep in [0, 1]
        """
        # We use (1-cos)/2 formula to get a 0->1 curve
        # Then scale it to min_beta->max_beta
        cosine_value = torch.cos(torch.tensor(t) * math.pi / 2)
        flipped_value = (1.0 - cosine_value ** 2)  # This gives us 0->1
        
        # Scale to our sigma range
        sigma = self.min_beta + (self.max_beta - self.min_beta) * flipped_value
        
        # Apply shift parameter for compatibility with discrete flow
        if self.shift != 1.0:
            sigma = (self.shift * sigma) / (1 + (self.shift - 1) * sigma)
            
        return sigma
    
    @property
    def sigma_min(self):
        return self.sigmas[0]  # Now this is the minimum (at the beginning)
    
    @property
    def sigma_max(self):
        return self.sigmas[-1]  # Now this is the maximum (at the end)
    
    def get_schedule(self, num_steps):
        """Return the full sigma schedule"""
        req_sigmas = self.sigmas.clone()[:num_steps]
        return req_sigmas
    
    def timestep(self, sigma):
        """Convert sigma to timestep value"""
        return sigma * 1000
    
    def sigma(self, timestep: torch.Tensor):
        """
        Get sigma for a given timestep index
        timestep: in range [1, num_steps]
        """
        # Convert to tensor if not already
        if not torch.is_tensor(timestep):
            timestep = torch.tensor(timestep, device=self.sigmas.device)
            
        # Handle batched timesteps
        is_scalar = timestep.ndim == 0
        if is_scalar:
            timestep = timestep.unsqueeze(0)
            
        # Convert to indices and clamp to valid range
        # Subtract 1 because the input is 1-indexed but our array is 0-indexed
        indices = (timestep - 1).long().clamp(0, self.num_steps - 1)
        result = self.sigmas[indices]
        
        # Return scalar if input was scalar
        if is_scalar:
            return result.squeeze(0)
        return result
    
    def calculate_denoised(self, sigma, model_output, model_input):
        """
        Calculate denoised x0 from model output
        Uses exact same formula as ModelSamplingDiscreteFlow
        """
        sigma = sigma.view(sigma.shape[:1] + (1,) * (model_output.ndim - 1))
        return model_input - model_output * sigma
    
    def noise_scaling(self, sigma, noise, latent_image, max_denoise=False):
        """
        Apply noise scaling for next step
        Uses exact same formula as ModelSamplingDiscreteFlow
        """
        return sigma * noise + (1.0 - sigma) * latent_image
    

if __name__ == "__main__":
    scheduler_karras = KarrasScheduler(num_steps=30, rho=2)
    scheduler_exp = ExponentialScheduler(num_steps=1000)
    scheduler_warmcosine = WarmupCosineScheduler(num_steps=1000, warmup_steps=10)
    scheduler = ModelSamplingDiscreteFlow(shift=5)

    sigmas_karras = scheduler_karras.get_schedule()
    sigmas_exp = scheduler_exp.get_schedule(num_steps=30)
    sigmas_warmcosine = scheduler_warmcosine.get_schedule(num_steps=30)
    sigmas = scheduler.get_schedule(num_steps=30)

    plt.plot(sigmas_karras.cpu().numpy(), label='Karras Scheduler')
    plt.plot(sigmas_exp.cpu().numpy(), label='Exponential Scheduler')
    # plt.plot(sigmas_warmcosine.cpu().numpy(), label='Warm Cosine Scheduler')
    plt.plot(sigmas.cpu().numpy(), label='Discrete Flow Scheduler')

    # Add labels and legend
    plt.title("Noise Schedules Comparison")
    plt.xlabel("Timestep")
    plt.ylabel("Sigma (Noise Level)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("noise_schedule_comparison.png")