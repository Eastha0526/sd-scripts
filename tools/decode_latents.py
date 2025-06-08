import argparse
import os
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import CLIPTokenizer
from diffusers import EulerDiscreteScheduler
from safetensors.torch import load_file

# Assuming these are in the same directory or in the Python path
from library import model_util, sdxl_model_util
import networks.lora as lora
from library.utils import setup_logging
from library.device_utils import init_ipex, get_preferred_device

setup_logging()
import logging

logger = logging.getLogger(__name__)

# Constants
SCHEDULER_LINEAR_START = 0.00085
SCHEDULER_LINEAR_END = 0.0120
SCHEDULER_TIMESTEPS = 1000
SCHEDLER_SCHEDULE = "scaled_linear"

def init_ipex():
    # Placeholder for init_ipex function
    pass

def get_preferred_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_npz_file(npz_path, device, dtype):
    data = np.load(npz_path)
    latents = torch.from_numpy(data['latents']).to(device, dtype=dtype)
    original_size = data['original_size']
    crop_ltrb = data['crop_ltrb']
    return latents, original_size, crop_ltrb

if __name__ == "__main__":
    init_ipex()

    DEVICE = get_preferred_device()
    DTYPE = torch.float16

    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--npz_dir", type=str, required=True, help="Directory containing .npz files")
    parser.add_argument("--output_dir", type=str, default=".")
    args = parser.parse_args()

    # Load models
    text_model1, text_model2, vae, unet, _, _ = sdxl_model_util.load_models_from_sdxl_checkpoint(
        sdxl_model_util.MODEL_VERSION_SDXL_BASE_V1_0, args.ckpt_path, "cpu"
    )

    unet.to(DEVICE, dtype=DTYPE)
    unet.eval()

    vae_dtype = torch.float32 if DTYPE == torch.float16 else DTYPE
    vae.to(DEVICE, dtype=vae_dtype)
    vae.eval()
    vae.use_slicing = True

    # Process all .npz files in the specified directory
    for npz_file in tqdm(os.listdir(args.npz_dir)):
        if npz_file.endswith('.npz'):
            npz_path = os.path.join(args.npz_dir, npz_file)
            latents, original_size, crop_ltrb = load_npz_file(npz_path, DEVICE, DTYPE)
            latents = latents.unsqueeze(0)
            # Decode latents
            with torch.no_grad():
                latents = latents.to(vae_dtype)
                image = vae.decode(latents).sample
                image = (image / 2 + 0.5).clamp(0, 1)

            # Convert to PIL Image
            image = image.cpu().permute(0, 2, 3, 1).float().numpy()
            image = (image * 255).round().astype("uint8")
            pil_image = Image.fromarray(image[0])

            # Save as WebP
            output_filename = os.path.splitext(npz_file)[0] + ".webp"
            output_path = os.path.join(args.output_dir, output_filename)
            pil_image.save(output_path, format="WebP")

            print(f"Processed {npz_file} and saved as {output_filename}")

    print("Done!")