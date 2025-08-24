import asyncio
import time
from datetime import datetime
import aiohttp
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
import torch
import json
from torchmetrics.image.fid import FrechetInceptionDistance

# Import FID calculation functions

class CacheBenchmarkClient:
    def __init__(self):
        self.server_url = "http://localhost:8000"
        self.poll_interval = 0.1
        self.empty_pool_timeout = 10000
        self.output_dir = "cache_benchmark_results"
        self.metrics_file = os.path.join(self.output_dir, "metrics.json")
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"Created output directory: {self.output_dir}")
        # Initialize or load existing metrics file
        if os.path.exists(self.metrics_file):
            with open(self.metrics_file, 'r') as f:
                self.metrics = json.load(f)
        else:
            self.metrics = {
                'benchmark_start_time': datetime.now().isoformat(),
                'prompts': {}
            }

    async def change_cache_interval(self, interval):
        url = f"{self.server_url}/change_caching_interval"
        params = {"cache_interval": interval}
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, params=params) as response:
                if response.status == 200:
                    print(f"Successfully changed cache interval to {interval}")
                    return await response.json()
                else:
                    error_message = await response.text()
                    print(f"Failed to change cache interval: {error_message}")
                    raise Exception(f"Failed to change cache interval: {error_message}")

    async def add_request(self, prompt, timesteps_left):
        url = f"{self.server_url}/add_request"
        data = {"prompt": prompt, "timesteps_left": timesteps_left}

        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=data) as response:
                if response.status == 200:
                    print(f"Added request with prompt: {prompt[:50]}...")
                    return await response.json()
                else:
                    error_message = await response.text()
                    print(f"Failed to add request: {error_message}")
                    raise Exception(f"Failed to add request: {error_message}")

    async def get_output(self):
        url = f"{self.server_url}/get_output"
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status == 200:
                    result = await response.json()
                    return result.get("completed_requests", [])
                else:
                    error_message = await response.text()
                    print(f"Failed to get output: {error_message}")
                    raise Exception(f"Failed to get output: {error_message}")

    async def start_background_process(self):
        url = f"{self.server_url}/start_background_process"
        params = {"model": ""}
        async with aiohttp.ClientSession() as session:
            async with session.post(url, params=params) as response:
                if response.status == 200:
                    print("Successfully started background process")
                    return await response.json()
                else:
                    error_message = await response.text()
                    print(f"Failed to start background process: {error_message}")
                    raise Exception(f"Failed to start background process: {error_message}")

    async def poll_with_timeout(self, num_requests):
        start_empty_time = None
        completed_requests = []
        print(f"Waiting for {num_requests} request(s) to complete...")

        while len(completed_requests) < num_requests:
            outputs = await self.get_output()
            
            if outputs:
                for output in outputs:
                    if output not in completed_requests:
                        completed_requests.append(output)
                        print(f"Received result {len(completed_requests)}/{num_requests}")
                start_empty_time = None
            else:
                if start_empty_time is None:
                    start_empty_time = time.time()
                elif time.time() - start_empty_time > self.empty_pool_timeout:
                    print("Timeout reached while waiting for results")
                    return completed_requests

            if len(completed_requests) < num_requests:
                await asyncio.sleep(self.poll_interval)

        return completed_requests

    def reset_fid(self):
        """Reset FID metric for new calculation"""
        self.fid = FrechetInceptionDistance(feature=64)

    def calculate_fid(self, generated_image_path, prompt_index):
        """Calculate FID score between generated image and references"""
        try:
            reference_folder = os.path.join("/home/fast-dit-serving/assets/fid", f"prompt_{prompt_index}")
            
            # Reset FID metric
            self.reset_fid()
            
            # Load and process reference images
            reference_tensors = []
            for ref_img_name in os.listdir(reference_folder):
                if ref_img_name.endswith(('.png', '.jpg', '.jpeg')):
                    ref_img_path = os.path.join(reference_folder, ref_img_name)
                    ref_img = Image.open(ref_img_path).convert('RGB')
                    ref_tensor = torch.from_numpy(np.array(ref_img)).permute(2, 0, 1).unsqueeze(0)
                    reference_tensors.append(ref_tensor)
                
            
            # Concatenate all reference images into a single batch
            if reference_tensors:
                reference_batch = torch.cat(reference_tensors, dim=0)
                self.fid.update(reference_batch, real=True)
            else:
                print(f"No reference images found in {reference_folder}")
                return None
            
            # Load and process generated image
            gen_img = Image.open(generated_image_path).convert('RGB')
            gen_tensor = torch.from_numpy(np.array(gen_img)).permute(2, 0, 1).unsqueeze(0)
            gen_batch = gen_tensor.repeat(2, 1, 1, 1)
            self.fid.update(gen_batch, real=False)
            
            # Calculate FID
            fid_value = self.fid.compute().item()
            
            # Check for NaN or Inf
            if not np.isfinite(fid_value):
                print("Warning: FID calculation resulted in NaN or Inf")
                return None
                
            return float(fid_value)
            
        except Exception as e:
            print(f"Error calculating FID: {e}")
            return None

    def create_grid_image(self, image_paths, prompts, cache_interval):
        """Create a grid of images with their prompts"""
        num_images = len(image_paths)
        if num_images == 0:
            print("No images to create grid from")
            return None

        print(f"Creating grid image for {num_images} images...")
        
        # Load all images
        images = [Image.open(path) for path in image_paths]
        
        # Get dimensions
        img_width = images[0].width
        img_height = images[0].height
        
        # Calculate grid dimensions
        grid_cols = min(3, num_images)
        grid_rows = (num_images + grid_cols - 1) // grid_cols
        
        # Create blank image
        margin = 50  # Space for text
        grid_width = grid_cols * img_width
        grid_height = grid_rows * (img_height + margin)
        
        # Add title space
        title_height = 100
        result = Image.new('RGB', (grid_width, grid_height + title_height), 'white')
        draw = ImageDraw.Draw(result)
        
        # Add title
        title = f"Cache Interval: {cache_interval}"
        draw.text((20, 20), title, fill='black')
        
        # Paste images and add prompts
        for idx, (img, prompt) in enumerate(zip(images, prompts)):
            row = (idx // grid_cols)
            col = idx % grid_cols
            x = col * img_width
            y = row * (img_height + margin) + title_height
            
            # Paste image
            result.paste(img, (x, y))
            
            # Add prompt text
            text_y = y + img_height + 5
            draw.text((x + 5, text_y), prompt[:50] + "..." if len(prompt) > 50 else prompt, fill='black')
        
        print(f"Grid image created successfully")
        return result

    def log_metrics(self, cache_interval, prompt, fid_score, image_path):
        """Log metrics to the JSON metrics file"""
        if prompt not in self.metrics['prompts']:
            self.metrics['prompts'][prompt] = {}
        
        self.metrics['prompts'][prompt][f'cache_{cache_interval}'] = {
            'fid_score': fid_score,
            'image_path': image_path
        }
        
        # Save updated metrics to file
        with open(self.metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=4)

    async def run_benchmark(self, prompts, timesteps=30):
        """Run benchmark for different cache intervals"""
        try:
            print("\n=== Starting Benchmark ===")
            print("Initializing background process...")
            await self.start_background_process()
            
            results = {}
            
            # Test each cache interval
            for interval in range(1, 4):
                print(f"\n--- Testing Cache Interval: {interval} ---")
                
                # Change cache interval
                await self.change_cache_interval(interval)
                
                # Process prompts sequentially
                completed_requests = []
                for i, prompt in enumerate(prompts, 1):
                    print(f"\nProcessing prompt {i}/{len(prompts)}")
                    # Submit one request
                    await self.add_request(prompt, timesteps)
                    
                    # Wait for this specific request to complete
                    result = await self.poll_with_timeout(1)
                    if result:
                        completed_requests.extend(result)
                        print(f"Successfully completed prompt {i}")
                        
                        # Calculate FID for the generated image
                        image_path = result[0]["image"]
                        print(f"Calculating FID score for image: {image_path}")
                        fid = self.calculate_fid(image_path, i-1)
                        
                        # Log metrics
                        if fid is not None:
                            print(f"FID Score: {fid:.2f}")
                            self.log_metrics(interval, prompt, fid, image_path)
                        else:
                            print("Failed to calculate FID score")
                    else:
                        print(f"Failed to complete prompt {i}")
                    
                    # Wait a bit before next request
                    await asyncio.sleep(1)
                
                if completed_requests:
                    print(f"\nCreating visualization for cache interval {interval}")
                    # Create grid image
                    image_paths = [req["image"] for req in completed_requests]
                    prompts_used = [req["prompt"] for req in completed_requests]
                    
                    grid_image = self.create_grid_image(image_paths, prompts_used, interval)
                    if grid_image:
                        output_path = os.path.join(self.output_dir, f"cache_interval_{interval}.png")
                        grid_image.save(output_path)
                        print(f"Saved grid image: {output_path}")
                        
                        results[interval] = {
                            "grid_image": output_path,
                            "completed_requests": completed_requests
                        }
                
                print(f"Completed testing cache interval {interval}")
                await asyncio.sleep(2)  # Cool down between intervals
            
            print("\n=== Benchmark Completed ===")
            
            # Add final benchmark stats
            self.metrics['benchmark_end_time'] = datetime.now().isoformat()
            self.metrics['total_prompts'] = len(prompts)
            self.metrics['cache_intervals_tested'] = list(range(1, 11))
            
            # Save final metrics
            with open(self.metrics_file, 'w') as f:
                json.dump(self.metrics, f, indent=4)
            
            print(f"Metrics have been saved to: {self.metrics_file}")
            return results

        except Exception as e:
            print(f"\nError during benchmark: {e}")
            return None


async def main():
    # Test prompts
    prompts = [
        "A vast landscape made entirely of various meats spreads out before the viewer. tender, succulent hills of roast beef, chicken drumstick trees, bacon rivers, and ham boulders create a surreal, yet appetizing scene. the sky is adorned with pepperoni sun and salami clouds",
        "A silhouette of a grand piano overlooking a dusky cityscape viewed from a top-floor penthouse, rendered in the bold and vivid sytle of a vintage travel poster",
        '''stars, water, brilliantly
gorgeous large scale scene,
a little girl, in the style of
dreamy realism, light gold
and amber, blue and pink,
brilliantly illuminated in the
background''',
         '''Pirate ship trapped in a
cosmic maelstrom nebula,
rendered in cosmic beach
whirlpool engine,
volumetric lighting,
spectacular, ambient lights,
light pollution, cinematic
atmosphere, art nouveau
style, illustration art artwork
by SenseiJaye, intricate
detail''', 
'''colored sketch in the style of ck-ccd, young Asian woman wearing a motorcycle helmet, long loose platinum hair, sitting on a large powerful motorcycle, leather jacket, sunset, in orange hues'''
        # "A painter study hard to learn how to draw with many concepts in the air, white background",
        # "8k uhd A man looks up at the starry sky, lonely and ethereal, Minimalism, Chaotic composition Op Art"
    ]

    print("\nStarting Cache Interval Benchmark")
    print(f"Number of test prompts: {len(prompts)}")
    client = CacheBenchmarkClient()
    results = await client.run_benchmark(prompts)
    
    if results:
        print("\nBenchmark Results Summary:")
        for interval, data in results.items():
            print(f"Cache Interval {interval}: {data['grid_image']}")
    else:
        print("\nBenchmark failed to complete")

if __name__ == "__main__":
    asyncio.run(main())