import argparse
from pathlib import Path

from diffusers import DiffusionPipeline


OUTPUT_DIR = Path("/tmp/OUT")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate images using a trained diffusion model"
    )
    parser.add_argument(
        "-m",
        "--model_path",
        type=str,
        default="/home/sh/o/unsymbols/r/diffusers/examples/unconditional_image_generation/out_sym_lg/",
        help="Path to the trained model directory",
    )
    parser.add_argument(
        "-l",
        "--num_images",
        type=int,
        default=10,
        help="Number of images to generate",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    model_path = str(Path(args.model_path))

    pipeline = DiffusionPipeline().from_pretrained(model_path).to("cuda")
    for i in range(args.num_images):
        print(f"Generating image {i+1}/{args.num_images}")
        image = pipeline(num_inference_steps=100).images[0]
        image.save(OUTPUT_DIR / f"image_{i}.png")


if __name__ == "__main__":
    main()
