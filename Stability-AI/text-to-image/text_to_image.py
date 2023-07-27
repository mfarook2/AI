import streamlit as st
import os
import requests
import base64
from PIL import Image
import datetime

#Stability AI's RST API call for text-to-image
def text_to_image(api_host, api_key, engine_id, height, width, prompt, \
 weight, cfg_scale, clip_guidance_preset, sampler, seed, steps, style_preset):
    response = requests.post(
        f"{api_host}/v1/generation/{engine_id}/text-to-image",
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
                "Authorization": f"Bearer {api_key}"
            },
            json={
                "text_prompts": [
                    {
                        "text": prompt
                    }
                ],
                "cfg_scale": 7,
                "clip_guidance_preset": clip_guidance_preset,
                "height": height,
                "width": width,
                "style_preset": style_preset,
                "steps": steps,
            },
        )

    if response.status_code != 200:
        raise Exception("Non-200 response: " + str(response.text))

    #images are stored in "out" folder
    if not os.path.exists('out'):
        os.makedirs('out')

    data = response.json()
    for i, image in enumerate(data["artifacts"]):
        timestamp = str(int(round(datetime.datetime.now().timestamp())))
        with open(f"./out/{timestamp}_{style_preset}_{i}.png", "wb") as f:
            f.write(base64.b64decode(image["base64"]))
            
    #print all the png files from out directory
    file_list = os.listdir("out")
    for file_name in file_list:
        file_path = os.path.join("out", file_name)
        image = Image.open(file_path)
        st.image(image)



#create the canvas for input parameters 
st.sidebar.title(':white[Stability AI text-to-image API Fine Tuning]')
tab1, tab2, tab3 = st.sidebar.tabs(["**Stability API Key**",  "**Parameters**", "**Information**"])

# API Key Header (tab1)
stability_api_key = tab1.text_input("Stability API Key")

#Parameters (tab 2)
engine_id = tab2.selectbox(
    'Engine ID',
    ('stable-diffusion-v1-5','stable-diffusion-xl-1024-v0-9','stable-diffusion-v1', 'stable-diffusion-v1-5', 'stable-diffusion-512-v2-0',
    'stable-diffusion-768-v2-0', 'stable-diffusion-depth-v2-0', 'stable-diffusion-512-v2-1','stable-diffusion-768-v2-1',
    'stable-diffusion-xl-beta-v2-2-2' ), help='Chose the engine ID')
style_preset=tab2.selectbox(
    'Style Preset',
    ('3d-model','analog-film','anime','cinematic','comic-book','digital-art','fantasy-art','isometric','line-art','low-poly' \
     'modeling-compound','neon-punk', 'origami','photographic','pixel-art', 'tile-texture'))
prompt = tab2.text_input('Prompt', max_chars=2000)
weight = tab2.number_input('Weight')
seed = tab2.slider('Seed',value=1, min_value=1,max_value=4294967295)
steps = tab2.slider('Steps',value=50,min_value=10,max_value=150)
cfg_scale = tab2.slider('CFG Scale', min_value=0, max_value=35, value=7)
clip_guidance_preset=tab2.selectbox(
    'Clip Guidance Preset',
    ('NONE','FAST_BLUE','FAST_GREEN','SIMPLE','SLOW','SLOWER','SLOWEST'))
sampler=tab2.selectbox(
    'Sampler',
    ('DDIM','DDPM','K_DPMPP_2M','K_DPMPP_2S_ANCESTRAL','K_DPM_2','K_DPM_2_ANCESTRAL','K_DPM_2_ANCESTRAL','K_DPM_2_ANCESTRAL','K_HEUN','K_LMS'))
height = tab2.number_input(min_value=128, step=64, label='Height', value=512)
width = tab2.number_input(min_value=128, step=64, label='Width', value=512)
total_area = height * width
extras = st.empty()

#Information (tab 3)
tab3.write("Rest API documentation:   \
 https://platform.stability.ai/docs/api-reference#tag/v1generation/operation/textToImage")

#setup the environment for stability api call.
api_host = os.getenv('API_HOST', 'https://api.stability.ai')
api_key = os.getenv("STABILITY_API_KEY")

#setup the environment for stability api call.
api_host = os.getenv('API_HOST', 'https://api.stability.ai')
api_key=None
api_key = os.getenv("STABILITY_API_KEY")
if (api_key is None):
    # API Key Header (tab1)
    api_key = stability_api_key


#generate the image
result = tab2.button('Submit')
if  result:
    if api_key is None:
        raise Exception("Missing Stability API key.")

    result = text_to_image(api_host,
                           api_key,
                           engine_id,
                           height,
                           width,
                           prompt,
                           weight,
                           cfg_scale,
                           clip_guidance_preset,
                           sampler,
                           seed,
                           steps,
                           style_preset,
                           )