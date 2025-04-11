import os
import gradio as gr
import torch
import numpy as np
from PIL import Image
from diffusers import (
    StableDiffusionControlNetPipeline, 
    StableDiffusion3InpaintPipeline,
    ControlNetModel, 
    EulerAncestralDiscreteScheduler,
    AutoPipelineForImage2Image,
    AutoPipelineForInpainting
)
from diffusers.utils import load_image
from controlnet_aux import HEDdetector
from accelerate import Accelerator
from diffusers import SD3Transformer2DModel, BitsAndBytesConfig

# Accelerator 인스턴스 생성
accelerator = Accelerator()

# CUDA 환경 설정 (필요 시)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# 모델 설정 딕셔너리 (확장 가능)
MODEL_CONFIGS = {
    "Ghibli-Diffusion": {
        "sd_model": "nitrosocke/Ghibli-Diffusion",
        "controlnet_model": "vsanimator/sketch-a-sketch",
        "scheduler": EulerAncestralDiscreteScheduler,
        "type": "controlnet"
    },
    "Stable-Diffusion-3.5-Large": {
        "sd_model": "stabilityai/stable-diffusion-3.5-large",  # transformer 기반 모델 사용
        "type": "img2img",
    },
    "Stable-Diffusion-3.5-Large-Inpainting": {
        "sd_model": "stabilityai/stable-diffusion-3.5-large",  # transformer 기반 모델 사용
        "type": "inpainting",
    }
}

class ModelManager:
    def __init__(self, default_model="Ghibli-Diffusion"):
        # 기본적으로 사용 가능한 GPU 할당 (필요에 따라 device 분산 가능)
        self.device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
        self.models = {}
        self.current_model = default_model
        self.hed = HEDdetector.from_pretrained('lllyasviel/Annotators')
        self.load_model(default_model)

    def load_model(self, model_name):
        if model_name in self.models:
            self.current_model = model_name
            return self.models[model_name]

        config = MODEL_CONFIGS[model_name]
        model_type = config["type"]

        if model_type == "controlnet":
            # ControlNet 모델 로드
            controlnet = ControlNetModel.from_pretrained(
                config["controlnet_model"], torch_dtype=torch.float16
            ).to(self.device)
            pipe = StableDiffusionControlNetPipeline.from_pretrained(
                config["sd_model"], controlnet=controlnet, torch_dtype=torch.float16
            ).to(self.device)
            pipe.safety_checker = None
            if "scheduler" in config:
                pipe.scheduler = config["scheduler"].from_config(pipe.scheduler.config)
        elif model_type == "img2img":
            # 이미지-투-이미지: 별도의 transformer 모델 로드 후 accelerator로 준비
            device_img2img = torch.device('cuda:1' if torch.cuda.device_count() > 1 else self.device)
            pipe = AutoPipelineForImage2Image.from_pretrained(
                config["sd_model"],
                torch_dtype=torch.float16
            ).to(device_img2img)
        elif model_type == "inpainting":
            # 인페인팅: 별도의 transformer 모델 로드 후 accelerator로 준비
            device_inpaint = torch.device('cuda:1' if torch.cuda.device_count() > 2 else self.device)
            pipe = AutoPipelineForInpainting.from_pretrained(
                config["sd_model"],
                torch_dtype=torch.float16
            ).to(device_inpaint)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        self.models[model_name] = pipe
        self.current_model = model_name
        return pipe

    def get_current_pipe(self):
        return self.models[self.current_model]

# ModelManager 인스턴스 생성 (기본 모델은 "Ghibli-Diffusion")
model_manager = ModelManager(default_model="Ghibli-Diffusion")

# 기본 설정
negative_prompt = ""
num_images = 6
threshold = 250

# 이미지 전처리 함수 (composite 미리보기용)
def predict(im):
    comp = im["composite"]
    pil_img = Image.fromarray(comp.astype('uint8')).resize((512, 512))
    pil_img = pil_img.convert("RGBA")
    bg = Image.new("RGBA", pil_img.size, (255, 255, 255, 255))
    bg.paste(pil_img, (0, 0), pil_img)
    final_img = bg.convert("RGB")
    return np.array(final_img)

def mask(im):
    comp = im["background"]
    return im['layers'][0]

# 스케치 함수: 현재 컨트롤넷 파이프라인을 사용하여 이미지 생성
def sketch(composite, prompt, negative_prompt, seed, num_steps):
    print("Sketching")
    pipe = model_manager.load_model("Ghibli-Diffusion")

    if composite is None:
        composite = np.full((512, 512, 3), 255, dtype=np.uint8)
    generator = torch.Generator(device=model_manager.device)
    generator.manual_seed(seed)
    composite_image = Image.fromarray(composite.astype(np.uint8)).convert("L")
    processed_image = composite_image.convert("RGB").point(lambda p: 255 if p > 128 else 0)
    # model_manager로부터 현재 파이프(예: controlnet 파이프)를 가져옴
    # pipe = model_manager.get_current_pipe()
    images = pipe(
        prompt, 
        processed_image, 
        negative_prompt=negative_prompt, 
        num_inference_steps=num_steps, 
        generator=generator, 
        controlnet_conditioning_scale=1.0
    ).images
    return images[0]

def run_sketching(prompt, composite, sketch_states, shadow_draw):
    to_return = []
    curr_sketch = composite
    for k in range(num_images):
        seed = sketch_states[k][1]
        if seed is None:
            seed = np.random.randint(1000)
            sketch_states[k][1] = seed
        new_image = sketch(curr_sketch, prompt, negative_prompt, seed=seed, num_steps=20)
        to_return.append(new_image)
    if shadow_draw:
        hed = model_manager.hed
        hed_images = [hed(image, scribble=False) for image in to_return]
        avg_hed = np.mean([np.array(image) for image in hed_images], axis=0)
        curr_sketch = Image.fromarray(np.uint8((1.0 * (1. - (avg_hed / 255.))) * 255.))
    else:
        curr_sketch = None
    return to_return + [curr_sketch, curr_sketch, sketch_states]

def select_image(image):
    if isinstance(image, (list, tuple)):
        return image[0]
    return image

def generate_img2img(selected_img, img2img_prompt, img2img_strength, img2img_guidance, img2img_negative_prompt):
    if selected_img is None:
        return None
    init_image = Image.fromarray(selected_img.astype('uint8')).resize((1024,1024))
    # 모델 전환: "Stable-Diffusion-3.5-Large" 모델 사용
    model_manager.load_model("Stable-Diffusion-3.5-Large")
    pipe = model_manager.get_current_pipe()
    result = pipe(
        prompt=img2img_prompt,
        image=init_image,
        strength=img2img_strength,
        guidance_scale=img2img_guidance,
        negative_prompt=img2img_negative_prompt,
        num_inference_steps=50
    )
    return result.images[0]

def generate_inpainted_image(inpaint_data, prompt, negative_prompt, strength, guidance_scale):
    if inpaint_data is None or "background" not in inpaint_data or len(inpaint_data.get("layers", [])) == 0:
        return None
    bg = Image.fromarray(inpaint_data["background"]).convert("RGB")
    mask_layer = Image.fromarray(inpaint_data["layers"][0].astype('uint8')).resize(bg.size)
    mask_layer = mask_layer.convert("RGBA")
    white_bg = Image.new("RGBA", mask_layer.size, (255, 255, 255, 255))
    white_bg.paste(mask_layer, (0, 0), mask_layer)
    mask_image = white_bg.convert("L")
    mask_image = Image.eval(mask_image, lambda x: 255 - x)
    mask_image = mask_image.point(lambda p: 255 if p > 128 else 0)
    # 모델 전환: "Stable-Diffusion-3.5-Large-Inpainting" 모델 사용
    model_manager.load_model("Stable-Diffusion-3.5-Large-Inpainting")
    pipe = model_manager.get_current_pipe()
    result = pipe(
        prompt=prompt,
        image=bg,
        mask_image=mask_image,
        strength=strength,
        width=1024,
        height=1024,
        guidance_scale=guidance_scale,
        negative_prompt=negative_prompt,
        num_inference_steps=50
    )
    return result.images[0]

def reset(sketch_states):
    for k in range(num_images):
        sketch_states[k] = [None, None]
    return None, None, sketch_states

# Gradio 인터페이스 구성
with gr.Blocks() as demo:
    start_state = [[None, None] for _ in range(num_images)]
    sketch_states = gr.State(start_state)
    checkbox_state = gr.State(True)
    white_brush = gr.Brush(default_color='#FFFFFF', colors=['#FFFFFF'], color_mode='fixed')

    with gr.Row():
        with gr.Column(scale=1):
            with gr.Tabs():
                with gr.TabItem("Sketch"):
                    i = gr.Sketchpad(brush=gr.Brush(default_size=3), canvas_size=(1024,1024))
                with gr.TabItem("Suggested Lines"):
                    i_sketch = gr.Image()
            prompt_box = gr.Textbox(label="Prompt")
            with gr.Row():
                btn = gr.Button("Render")
                checkbox = gr.Checkbox(label="Generated suggested lines", value=True)
                btn2 = gr.Button("Reset")
            i_prev = gr.Image(label="Composite Preview", interactive=False)
        
        with gr.Column(scale=1):
            out_imgs = []
            select_btns = []
            for idx in range(num_images):
                with gr.Column():
                    out_img = gr.Image(label=f"Generated Image {idx+1}", interactive=False)
                    select_btn = gr.Button("Select", variant="secondary")
                out_imgs.append(out_img)
                select_btns.append(select_btn)
        
        with gr.Column():
            selected_image_display = gr.Image(label="Selected Image")
            img2img_prompt = gr.Textbox(label="Image-to-Image Prompt")
            img2img_negative_prompt = gr.Textbox(label="Negative Prompt")
            img2img_strength = gr.Slider(minimum=0, maximum=1, value=0.75, label="Strength")
            img2img_guidance = gr.Slider(minimum=1, maximum=10, value=7.5, label="Guidance Scale")
            img2img_btn = gr.Button("Generate Image-to-Image")
            img2img_output = gr.Image(type='pil', label="Image-to-Image Result")
            img2img_select_btn = gr.Button("Select for Inpainting", variant="secondary")

        with gr.Column():
            inpaint2img = gr.ImageEditor(label='Inpaint', interactive=True)
            inpaint2img_prompt = gr.Textbox(label='prompt')
            inpaint2img_negative_prompt = gr.Textbox(label='negative prompt')
            inpaint2img_strength = gr.Slider(minimum=0, maximum=1, value=1.0, label="strength")
            inpaint2img_guidance = gr.Slider(minimum=1, maximum=10, value=7.5, label="guidance scale")
            inpaint2img_btn = gr.Button("Generate Inpainting")
            inpaint2img_outputs = gr.Image(type="pil", interactive=False)

    i.change(predict, inputs=i, outputs=i_prev, show_progress="hidden")
    btn.click(
        run_sketching, 
        inputs=[prompt_box, i_prev, sketch_states, checkbox_state], 
        outputs=out_imgs + [i_sketch, i_prev, sketch_states]
    )
    for out_img, select_btn in zip(out_imgs, select_btns):
        select_btn.click(select_image, inputs=out_img, outputs=selected_image_display)
    img2img_btn.click(
        generate_img2img,
        inputs=[selected_image_display, img2img_prompt, img2img_strength, img2img_guidance, img2img_negative_prompt],
        outputs=img2img_output
    )
    img2img_select_btn.click(
        select_image,
        inputs=img2img_output,
        outputs=inpaint2img
    )
    inpaint2img_btn.click(
        generate_inpainted_image,
        inputs=[inpaint2img, inpaint2img_prompt, inpaint2img_negative_prompt, inpaint2img_strength, inpaint2img_guidance],
        outputs=inpaint2img_outputs
    )
    btn2.click(reset, inputs=sketch_states, outputs=[i, i_prev, sketch_states])
    checkbox.change(lambda x: x, inputs=[checkbox], outputs=[checkbox_state])

demo.launch(share=True)