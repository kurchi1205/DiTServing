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
        self.fid = FrechetInceptionDistance(feature=192)
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
            gen_img = gen_img.repeat(2, 1, 1, 1)
            self.fid.update(gen_img, real=False)
            fid_value = self.fid.compute().item()
            self._load_source_images()  # Reset after each calculation
            return fid_value if np.isfinite(fid_value) else None
        except Exception as e:
            print(f"Error calculating FID for {generated_image_path}: {e}")
            return None
    
    def calculate_fid_for_directories(self, target_dirs):
        """Compute FID for all images in multiple target directories."""
        fid_scores = {}
        for target_dir in target_dirs:
            dir_name = os.path.basename(target_dir)
            fid_scores[dir_name] = {}
            for img_name in os.listdir(target_dir):
                if img_name.endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(target_dir, img_name)
                    fid_value = self.calculate_fid(img_path)
                    if fid_value is not None:
                        fid_scores[dir_name][img_name] = fid_value
        return fid_scores
    
    def plot_fid_scores(self, fid_scores, save_path="fid_plot.png"):
        """Plot FID scores as a line plot for multiple directories and save the plot."""
        if not fid_scores:
            print("No valid FID scores to plot.")
            return
        
        plt.figure(figsize=(12, 6))
        for dir_name, scores in fid_scores.items():
            img_names = list(scores.keys())
            fid_values = list(scores.values())
            plt.plot(img_names, fid_values, marker='o', linestyle='-', alpha=0.7, label=dir_name)
        
        plt.xticks(rotation=45, ha='right')
        plt.xlabel("Generated Images")
        plt.ylabel("FID Score")
        plt.title("FID Scores for Generated Images Across Directories")
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.savefig(save_path, bbox_inches='tight')
        plt.show()


# Example usage:
fid_calc = FIDCalculator("/home/DiTServing/assets/fid/prompt_3")
# fid_scores = fid_calc.calculate_fid_for_directories(["/home/DiTServing/custom_serve/gen_images_no_cache_baby_1", "/home/DiTServing/custom_serve/gen_images", "/home/DiTServing/stability_sd3_infer/gen_images_baby_og"])
# fid_scores = fid_calc.calculate_fid_for_directories(["/home/DiTServing/custom_serve/gen_images_no_cache_ship", "/home/DiTServing/custom_serve/gen_images", "/home/DiTServing/stability_sd3_infer/gen_images_ship_og", "/home/DiTServing/custom_serve/gen_images_ship_iter_10"])
fid_scores = fid_calc.calculate_fid_for_directories(["/home/DiTServing/custom_serve/gen_images_no_cache_prompt_1", "/home/DiTServing/custom_serve/gen_images_prompt_1_new", "/home/DiTServing/stability_sd3_infer/gen_images_prompt_1_og", "/home/DiTServing/custom_serve/gen_images_prompt_1_iter_10"])
# fid_scores = fid_calc.calculate_fid_for_directories(["/home/DiTServing/custom_serve/gen_images_biker_no_cache", "/home/DiTServing/custom_serve/gen_images", "/home/DiTServing/stability_sd3_infer/gen_images_biker_og", "/home/DiTServing/custom_serve/gen_images_biker_iter_10"])
fid_calc.plot_fid_scores(fid_scores)
