import json
from datasets import load_dataset
from tqdm import tqdm

def read_data():
    # Load the dataset
    ds = load_dataset("nateraw/parti-prompts", split="train")
    prompts_json = {}
    challenges_json = {}
    for i, example in tqdm(enumerate(ds)):
        prompt_key = f"p2_prompt_{i}"
        prompts_json[prompt_key] = example["Prompt"]
        challenges_json[f"{prompt_key}_challenge"] = example["Challenge"]

    # Save to JSON files
    with open("parti_prompts.json", "w") as f:
        json.dump(prompts_json, f, indent=2)

    with open("parti_challenges.json", "w") as f:
        json.dump(challenges_json, f, indent=2)

if __name__=='__main__':
    read_data()