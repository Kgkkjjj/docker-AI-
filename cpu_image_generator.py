#!/usr/bin/env python3
"""Simple CPU-based image generator using Hugging Face Diffusers."""

import argparse
import random
import re
from diffusers import StableDiffusionPipeline
import torch


def augment_prompt(prompt: str, info_file: str | None, num_lines: int) -> str:
    """Return the prompt augmented with optional lines from a game info file."""
    if not info_file:
        return prompt

    try:
        with open(info_file, "r", encoding="utf-8") as f:
            all_lines = [line.strip() for line in f if line.strip()]
    except OSError as e:
        print(f"Warning: couldn't read {info_file}: {e}")
        return prompt

    # Collect lines that share words with the prompt
    words = re.findall(r"\w+", prompt.lower())
    matching = [l for l in all_lines if any(w in l.lower() for w in words)]
    candidates = matching or all_lines
    random.shuffle(candidates)
    selected = candidates[: max(1, num_lines)]
    augmented = prompt + " " + " ".join(selected)
    return augmented


def main():
    parser = argparse.ArgumentParser(description="Generate an image from a text prompt on CPU.")
    parser.add_argument("prompt", help="Text prompt for image generation")
    parser.add_argument("--output", default="output.png", help="Path to save the generated image")
    parser.add_argument("--model", default="stabilityai/stable-diffusion-2-1-base", help="Model identifier from Hugging Face")
    parser.add_argument("--info", default=None, help="Optional path to infamous_second_son_info.txt")
    parser.add_argument("--lines", type=int, default=5, help="Number of info lines to append to the prompt")
    args = parser.parse_args()

    pipe = StableDiffusionPipeline.from_pretrained(args.model, torch_dtype=torch.float32)
    pipe.to("cpu")

    final_prompt = augment_prompt(args.prompt, args.info, args.lines)
    image = pipe(final_prompt, num_inference_steps=50).images[0]
    image.save(args.output)
    print(f"Saved image to {args.output}")


if __name__ == "__main__":
    main()
