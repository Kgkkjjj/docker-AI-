# docker-AI-

This project includes a lightweight CPU-based image generator and a data file containing information about *Infamous Second Son*. The generator uses [Hugging Face Diffusers](https://github.com/huggingface/diffusers) and can run without a GPU, though generation will be slower than on hardware accelerated systems.

## Requirements
- Python 3.10+
- `torch` and `diffusers`

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install diffusers transformers accelerate
```

## Usage
Run the `cpu_image_generator.py` script with a text prompt. The script loads a model and executes inference on the CPU.

```bash
python cpu_image_generator.py "a scenic view of downtown Seattle"
```

The resulting image is saved to `output.png`.

## Infamous Second Son Data
The file `infamous_second_son_info.txt` contains 500 lines of facts about the game, including details about characters such as Brooke Augustine.
