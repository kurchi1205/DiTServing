import json
import matplotlib.pyplot as plt

# Load saved LPIPS results
with open("/home/DiTServing/outputs/niqe_scores_by_prompt.json", "r") as f:
    niqe_data = json.load(f)

# Convert interval strings to integers and sort
intervals = sorted({int(interval) for scores in niqe_data.values() for interval in scores})
intervals_str = [str(i) for i in intervals]

# Plot each prompt's FID vs cache interval
plt.figure(figsize=(12, 6))

for prompt_name, scores in niqe_data.items():
    y = [scores.get(str(interval), None) for interval in intervals]
    if any(v is not None for v in y):  # Skip if all are None
        plt.plot(intervals, y, marker='o', label=prompt_name)

plt.xlabel("Cache Interval")
plt.ylabel("NIQE Score")
plt.title("NIQE Score vs Cache Interval (per prompt)")
plt.legend(title="Prompt", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.tight_layout()
plt.savefig("/home/DiTServing/outputs/niqe_scores_by_prompt_plot.png")
