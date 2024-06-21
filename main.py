import streamlit as st
from diffusers import DiffusionPipeline, AutoencoderKL
import torch


# Load the model
@st.cache_resource
def load_model(lora_path, lora_name):
    vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
    pipe = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        vae=vae,
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True
    )
    pipe.load_lora_weights(lora_path, weight_name=lora_name)

    if torch.backends.mps.is_available():
        device = "mps"
    else: 
        device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe.to(device)

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
