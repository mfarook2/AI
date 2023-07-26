import os
import io
import streamlit as st
import warnings
from PIL import Image
from stability_sdk import client
import stability_sdk.interfaces.gooseai.generation.generation_pb2 as generation

#Setup the appl layout
#create the canvas for input parameters 
title = "Stability AI text-to-image API fine Tuning :color[blue]"
total_area_exception = RuntimeError('Height x width should be between 589824 and 1048576')

st.sidebar.title(':white[Stability AI text-to-image API fine Tuning]')

tab1, tab2, tab3 = st.sidebar.tabs(["**Stability API Key**", "**Header**", "**Body**"])

#TAB 1
stability_api_key = tab1.text_input("Stability API Key")

# Body
prompt = tab3.text_input('Prompt', max_chars=2000)
engine_id = tab3.selectbox(
    'Engine ID',
    ('stable-diffusion-v1-5','stable-diffusion-xl-1024-v0-9','stable-diffusion-v1', 'stable-diffusion-v1-5', 'stable-diffusion-512-v2-0',
    'stable-diffusion-768-v2-0', 'stable-diffusion-depth-v2-0', 'stable-diffusion-512-v2-1','stable-diffusion-768-v2-1',
    'stable-diffusion-xl-beta-v2-2-2' ), help='Chose the engine ID')
height = tab3.number_input(min_value=128, step=64, label='Height', value=512)
width = tab3.number_input(min_value=128, step=64, label='Width', value=512)
total_area = height * width
if  (589824 <= total_area <= 1048576):
    tab3.exception(total_area_exception)

weight = tab3.number_input('Weight')
#cfg_scale = tab3.number_input('CFG Scale', min_value=0, max_value=35, value=7)
cfg_scale = tab3.slider('CFG_SCALE',min_value=0, max_value=35, value=7)
clip_guidance_preset=tab3.selectbox(
    'Clip Guidance Preset',
    ('NONE','FAST_BLUE','FAST_GREEN','SIMPLE','SLOW','SLOWER','SLOWEST'))
sampler=tab3.selectbox(
    'Sampler',
    ('DDIM','DDPM','K_DPMPP_2M','K_DPMPP_2S_ANCESTRAL','K_DPM_2','K_DPM_2_ANCESTRAL','K_DPM_2_ANCESTRAL','K_DPM_2_ANCESTRAL','K_HEUN','K_LMS'))
seed = tab3.slider('Seed',value=1, min_value=1,max_value=4294967295)
steps = tab3.slider('Steps',value=50,min_value=10,max_value=150)
style_preset=tab3.selectbox(
    'Style Preset',
    ('3d-model','analog-film','anime','cinematic','comic-book','digital-art','fantasy-art','isometric','line-art','low-poly' \
     'modeling-compound','neon-punk', 'origami','photographic','pixel-art', 'tile-texture'))
extras = st.empty()

#generate the image
result = tab3.button('Submit')
    
# Set up our connection to the API.
stability_api = client.StabilityInference(
    key=os.environ['STABILITY_KEY'], # API Key reference.
    verbose=True, # Print debug messages.
    engine=engine_id, # Set the engine to use for generation.
    # Available engines: stable-diffusion-xl-1024-v0-9 stable-diffusion-v1 stable-diffusion-v1-5 stable-diffusion-512-v2-0 stable-diffusion-768-v2-0
    # stable-diffusion-512-v2-1 stable-diffusion-768-v2-1 stable-diffusion-xl-beta-v2-2-2 stable-inpainting-v1-0 stable-inpainting-512-v2-0
)

print ("Sampler:  ", 'generation' + '.' + sampler)
print ("seed:", seed)
result = stability_api.generate(
    prompt=prompt,
    seed=seed,
    steps=steps,
    height=height,
    width=width,
    #weight=weight,
    cfg_scale=cfg_scale,
    #clip_guidance_preset=clip_guidance_preset,
    #sampler='generation' + '.' + SAMPLER_K_DPMPP_2M,
    sampler=generation.SAMPLER_K_DPMPP_2M
)
'''
# Set up our initial generation parameters.
answers = stability_api.generate(
    prompt="expansive landscape rolling greens with blue daisies and yggdrasil under a blue alien sky, masterful, ghibli",
    seed=992446758, # If a seed is provided, the resulting generated image will be deterministic.
                    # What this means is that as long as all generation parameters remain the same, you can always recall the same image simply by generating it again.
                    # Note: This isn't quite the case for CLIP Guided generations, which we tackle in the CLIP Guidance documentation.
    steps=50, # Amount of inference steps performed on image generation. Defaults to 30.
    cfg_scale=8.0, # Influences how strongly your generation is guided to match your prompt.
                   # Setting this value higher increases the strength in which it tries to match your prompt.
                   # Defaults to 7.0 if not specified.
    width=1024, # Generation width, if not included defaults to 512 or 1024 depending on the engine.
    height=1024, # Generation height, if not included defaults to 512 or 1024 depending on the engine.
    samples=1, # Number of images to generate, defaults to 1 if not included.
    sampler=generation.SAMPLER_K_DPMPP_2M # Choose which sampler we want to denoise our generation with.
                                                 # Defaults to k_dpmpp_2m if not specified. Clip Guidance only supports ancestral samplers.
                                                 # (Available Samplers: ddim, plms, k_euler, k_euler_ancestral, k_heun, k_dpm_2, k_dpm_2_ancestral, k_dpmpp_2s_ancestral, k_lms, k_dpmpp_2m, k_dpmpp_sde)
)

#result = stability.generate()
''' 

for resp in result:
    for artifact in resp.artifacts:
        if artifact.finish_reason == generation.FILTER:
            warnings.warn(
                "Your request activated the API's safety filters and could not be processed."
                "Please modify the prompt and try again.")
        if artifact.type == generation.ARTIFACT_IMAGE:
            img = Image.open(io.BytesIO(artifact.binary))
            img.save(str(artifact.seed)+ ".png") # Save our generated images with their seed number as the filename.
           