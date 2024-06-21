import streamlit as st
from diffusers import DiffusionPipeline, AutoencoderKL
import torch


# Load the model
@st.cache_resource
def load_model(lora_path, lora_name):
    """
    Load and initialize the Stable Diffusion XL model with LoRA weights.

    Parameters:
    lora_path (str): The path to the LoRA weights file.
    lora_name (str): The name of the LoRA weights file.

    Returns:
    pipe (DiffusionPipeline): The initialized Stable Diffusion XL model with LoRA weights.

    The function initializes the Stable Diffusion XL model with LoRA weights,
    selects the appropriate device (MPS, CUDA, or CPU), and returns the initialized model.
    """

    # Load the Variational Autoencoder (VAE)
    vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)

    # Load the Stable Diffusion XL pipeline
    pipe = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        vae=vae,
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True
    )

    # Load the LoRA weights
    pipe.load_lora_weights(lora_path, weight_name=lora_name)

    # Select the appropriate device
    if torch.backends.mps.is_available():
        device = "mps"
    else: 
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Move the model to the selected device
    pipe.to(device)

    # Return the initialized model
    return pipe

pipe = load_model("statue_LoRA/pytorch_lora_weights.safetensors", "pytorch_lora_weights.safetensors")

# Streamlit interface
st.title("Stable DiffusionXL-LoRa Image Generator")

prompt = st.text_input("Enter your prompt here:")

if st.button("Generate Image"):
    if prompt:
        with st.spinner('Generating image...'):
            final_prompt = "a photo of CUS statue, {}".format(prompt)

            image = pipe(prompt=prompt, num_inference_steps=25).images[0]
            st.image(image, caption=f"Generated Image for: {prompt}")
    else:
        st.warning("Please enter a prompt.")
