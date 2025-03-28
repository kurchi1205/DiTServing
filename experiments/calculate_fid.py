import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from torchmetrics.image.fid import FrechetInceptionDistance

class FIDCalculator:
    def __init__(self, source_dir):
        """Initialize with the source directory containing reference images."""
        self.source_dir = source_dir
        self.fid = FrechetInceptionDistance(normalize=True)
        self._load_source_images()

    def _load_source_images(self):
        """Load all source images into the FID metric."""
        self.fid.reset()
        source_tensors = []
        for img_name in os.listdir(self.source_dir):
            if img_name.endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(self.source_dir, img_name)
                img = self._process_image(img_path)
                source_tensors.append(img)
        
        if source_tensors:
            source_batch = torch.cat(source_tensors, dim=0)
            self.fid.update(source_batch, real=True)
        else:
            raise ValueError("No valid source images found.")
    
    def _process_image(self, image_path):
        """Load and preprocess an image."""
        img = Image.open(image_path).convert('RGB')
        img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).unsqueeze(0)
        return img_tensor
    
    def calculate_fid(self, generated_image_path):
        """Calculate FID score between a generated image and source images."""
        try:
            gen_img = self._process_image(generated_image_path)
            gen_img_batch = gen_img.repeat(2, 1, 1, 1)
            print(gen_img_batch.size())
            self.fid.update(gen_img_batch, real=False)
            fid_value = self.fid.compute().item()
            self._load_source_images()
            # self.fid.reset()  # Reset after each calculation
            return fid_value if np.isfinite(fid_value) else None
        except Exception as e:
            print(f"Error calculating FID for {generated_image_path}: {e}")
            return None
    
    def calculate_fid_for_directory(self, target_dir):
        """Compute FID for all images in the target directory."""
        fid_scores = {}
        for img_name in os.listdir(target_dir):
            if img_name.endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(target_dir, img_name)
                fid_value = self.calculate_fid(img_path)
                if fid_value is not None:
                    fid_scores[img_name] = fid_value
        return fid_scores
    
    def plot_fid_scores(self, fid_scores, save_path="fid_plot.png"):
        """Plot FID scores as a line plot and save the plot."""
        if not fid_scores:
            print("No valid FID scores to plot.")
            return
        
        img_names = list(fid_scores.keys())
        fid_values = list(fid_scores.values())
        
        plt.figure(figsize=(12, 6))
        plt.plot(img_names, fid_values, marker='o', linestyle='-', color='blue', alpha=0.7)
        plt.xticks(rotation=45, ha='right')
        plt.xlabel("Generated Images")
        plt.ylabel("FID Score")
        plt.title("FID Scores for Generated Images")
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.savefig(save_path, bbox_inches='tight')
        plt.show()


# Example usage:
fid_calc = FIDCalculator("/home/DiTServing/assets/fid/prompt_0")
fid_scores = fid_calc.calculate_fid_for_directory("/home/DiTServing/assets/fid_2/prompt_0")
fid_calc.plot_fid_scores(fid_scores)
