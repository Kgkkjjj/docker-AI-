#!/usr/bin/env python3
"""Simple CPU-based image generator using Hugging Face Diffusers."""

import argparse
from diffusers import StableDiffusionPipeline
import torch


def main():
    parser = argparse.ArgumentParser(description="Generate an image from a text prompt on CPU.")
    parser.add_argument("prompt", help="Text prompt for image generation")
    parser.add_argument("--output", default="output.png", help="Path to save the generated image")
    parser.add_argument("--model", default="stabilityai/stable-diffusion-2-1-base", help="Model identifier from Hugging Face")
    args = parser.parse_args()

    pipe = StableDiffusionPipeline.from_pretrained(args.model, torch_dtype=torch.float32)
    pipe.to("cpu")

    image = pipe(args.prompt, num_inference_steps=50).images[0]
    image.save(args.output)
    print(f"Saved image to {args.output}")


if __name__ == "__main__":
    main()
