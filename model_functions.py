import os
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
from rembg import remove

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

# Accelerator 인스턴스 생성
accelerator = Accelerator()
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

class ModelManager:
    def __init__(self, default_model="Ghibli-Diffusion"):
        # 기본적으로 사용 가능한 GPU 할당 (필요 시 device 분산 가능)
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
            device_img2img = torch.device('cuda:1' if torch.cuda.device_count() > 1 else self.device)
            pipe = AutoPipelineForImage2Image.from_pretrained(
                config["sd_model"],
                torch_dtype=torch.float16
            ).to(device_img2img)
        elif model_type == "inpainting":
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

def remove_bg(im):
    
    removed_bg = remove(im)
    # RGBA로 변환
    removed_bg = removed_bg.convert("RGBA")
    # 흰색 배경 이미지 생성
    white_bg = Image.new("RGBA", removed_bg.size, (255, 255, 255, 255))
    # 투명 영역이 있는 removed_bg를 흰색 배경 위에 합성 (mask 파라미터를 이용)
    final_image = Image.alpha_composite(white_bg, removed_bg)
    # 최종 결과를 RGB 모드로 변환
    final_image = final_image.convert("RGB")
    return final_image

# 이미지 전처리 함수 (composite 미리보기용)
def predict(im):
    comp = im["composite"]
    pil_img = Image.fromarray(comp.astype('uint8')).resize((512, 512))
    pil_img = pil_img.convert("RGBA")
    bg = Image.new("RGBA", pil_img.size, (255, 255, 255, 255))
    bg.paste(pil_img, (0, 0), pil_img)
    final_img = bg.convert("RGB")
    return np.array(final_img)

# 스케치 함수: 현재 컨트롤넷 파이프라인을 사용하여 이미지 생성
def sketch(composite, prompt, negative_prompt, remove_background, seed, num_steps):
    print("Sketching")
    pipe = model_manager.load_model("Ghibli-Diffusion")

    if composite is None:
        composite = np.full((512, 512, 3), 255, dtype=np.uint8)
    generator = torch.Generator(device=model_manager.device)
    generator.manual_seed(seed)
    composite_image = Image.fromarray(composite.astype(np.uint8)).convert("L")
    processed_image = composite_image.convert("RGB").point(lambda p: 255 if p > 128 else 0)
    images = pipe(
        prompt, 
        processed_image, 
        negative_prompt=negative_prompt, 
        num_inference_steps=num_steps, 
        generator=generator, 
        controlnet_conditioning_scale=1.0
    ).images
    # rembg를 사용해 배경 제거 (결과는 투명 배경을 가진 RGBA 이미지로 반환됨)
    if remove_background:
        return remove_bg(images[0])   
    return images[0]

def run_sketching(prompt, negative_prompt, composite, sketch_states, shadow_draw, remove_background,):
    to_return = []
    curr_sketch = composite
    for k in range(num_images):
        seed = sketch_states[k][1]
        if seed is None:
            seed = np.random.randint(1000)
            sketch_states[k][1] = seed
        new_image = sketch(curr_sketch, prompt, negative_prompt, remove_background, seed=seed, num_steps=20)
        to_return.append(new_image)
    if shadow_draw:
        hed = model_manager.hed
        hed_images = [hed(image, scribble=False) for image in to_return]
        avg_hed = np.mean([np.array(image) for image in hed_images], axis=0)
        curr_sketch = Image.fromarray(np.uint8((1.0 * (1. - (avg_hed / 255.))) * 255.))
    else:
        curr_sketch = None
    return to_return + [curr_sketch, curr_sketch, sketch_states]

def generate_img2img(selected_img, img2img_prompt, img2img_strength, img2img_guidance, img2img_negative_prompt, remove_background):
    if selected_img is None:
        return None
    init_image = Image.fromarray(selected_img.astype('uint8')).resize((1024,1024))
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
    if remove_background:
        return remove_bg(result.images[0])   
    return result.images[0]

def generate_inpainted_image(inpaint_data, prompt, negative_prompt, strength, guidance_scale, remove_background):
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
    if remove_background:
        return remove_bg(result.images[0])   
    return result.images[0]

def reset(sketch_states):
    for k in range(num_images):
        sketch_states[k] = [None, None]
    return None, None, sketch_states