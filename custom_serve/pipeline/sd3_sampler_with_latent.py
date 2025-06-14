import torch
import torch.nn as nn

class ModelSamplingDiscreteFlowAdaptive(nn.Module):
    """Discrete Flow Sampler with optional latent-based adaptive sigma scheduling"""
    
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
        start = self.timestep(self.sigmas[0])
        end = self.timestep(self.sigmas[-1])
        timesteps = torch.linspace(start, end, num_steps)
        sigs = []
        for x in range(len(timesteps)):
            ts = timesteps[x]
            sigs.append(self.sigma(ts))
        return torch.FloatTensor(sigs)

    def timestep(self, sigma):
        return sigma * 1000

    def sigma(self, timestep: torch.Tensor):
        # Standard schedule based on shift
        timestep = timestep / 1000.0
        if self.shift == 1.0:
            return timestep
        return (self.shift * (timestep ** 1.2)) / (1 + (self.shift - 1) * (timestep ** 1.2))

    def sigma_from_latent(self, latent: torch.Tensor, base_sigma: float, min_factor=0.8, max_factor=1.2):
        """
        Simplified version with just the most effective metrics.
        """
        # Primary metric: standard deviation (your original)
        std_complexity = latent.std(dim=(1, 2, 3), keepdim=False)
        
        # Secondary metric: edge content using simple gradients
        print(latent.size())
        print(torch.diff(latent, dim=-1)[..., :-1].pow(2).size())
        print(torch.diff(latent, dim=-2)[..., :-1].pow(2).size())

        grad_x = torch.diff(latent, dim=-1)  # Shape: [B, C, H, W-1]
        grad_y = torch.diff(latent, dim=-2)  # Shape: [B, C, H-1, W]
        
        # Crop to same size: take the overlapping region
        min_h = min(grad_x.shape[-2], grad_y.shape[-2])  # min(H, H-1) = H-1
        min_w = min(grad_x.shape[-1], grad_y.shape[-1])  # min(W-1, W) = W-1
        
        grad_x_cropped = grad_x[..., :min_h, :min_w]  # [B, C, H-1, W-1]
        grad_y_cropped = grad_y[..., :min_h, :min_w]  # [B, C, H-1, W-1]
        
        # Now they can be added
        grad_magnitude = torch.sqrt(grad_x_cropped.pow(2) + grad_y_cropped.pow(2))
        edge_complexity = grad_magnitude.mean(dim=(1, 2, 3))
        
        # Combine with weights
        combined = 0.6 * std_complexity + 0.4 * edge_complexity
        
        # Map to adjustment factor using known reference values
        # These are empirically determined "normal" complexity values
        reference_complexity = 0.8  # Typical complexity for medium-detail images
        normalized_complexity = combined / reference_complexity
        
        # Convert to adjustment factor
        adjustment = torch.sigmoid((normalized_complexity - 1.0) * 1.5)
        adjustment = min_factor + (max_factor - min_factor) * adjustment
        print("Adjustment: ", adjustment)
        return base_sigma * adjustment

    def calculate_denoised(self, sigma, model_output, model_input):
        sigma = sigma.view(sigma.shape[:1] + (1,) * (model_output.ndim - 1))
        return model_input - model_output * sigma

    def noise_scaling(self, sigma, noise, latent_image, max_denoise=False):
        """
        Combines latent and noise based on sigma.
        Now allows sigma to be a tensor for adaptive per-sample noise.
        """
        while sigma.ndim < latent_image.ndim:
            sigma = sigma.unsqueeze(-1)
        return sigma * noise + (1.0 - sigma) * latent_image
