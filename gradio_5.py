import gradio as gr
import torch
import numpy as np
import pandas as pd
from PIL import Image
from diffusers import (
    StableDiffusionControlNetPipeline, 
    StableDiffusion3InpaintPipeline,
    ControlNetModel, 
    EulerAncestralDiscreteScheduler,
    AutoPipelineForImage2Image,  # 이미지-투-이미지 파이프라인 임포트
    SD3Transformer2DModel,
    BitsAndBytesConfig,
)
from diffusers.utils import load_image
from controlnet_aux import HEDdetector
from accelerate import Accelerator
accelerator = Accelerator()
import os 
# 기본 설정
negative_prompt = ""
device = torch.device('cuda')  # controlnet 모델은 기본 cuda 사용 (예: cuda:0)
num_images = 3
threshold = 250  # 사용하지 않는 변수면 삭제 가능

# 모델 로드 (Sketch / ControlNet 모델)
controlnet = ControlNetModel.from_pretrained(
    "vsanimator/sketch-a-sketch", 
    torch_dtype=torch.float16
).to(device)

pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "nitrosocke/Ghibli-Diffusion", 
    controlnet=controlnet, 
    torch_dtype=torch.float16,
).to(device)
pipe.safety_checker = None
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

hed = HEDdetector.from_pretrained('lllyasviel/Annotators')

# 이미지-투-이미지 모델 로드 (cuda:1에서 동작)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
model_id = "stabilityai/stable-diffusion-3.5-medium"
nf4_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# load img2img model
device_img2img = torch.device("cuda:2")
model_nf4 = SD3Transformer2DModel.from_pretrained(
    model_id,
    subfolder="transformer",
    quantization_config=nf4_config,
    torch_dtype=torch.bfloat16
).to(device_img2img)

img2img_pipe = AutoPipelineForImage2Image.from_pretrained(
    model_id,
    transformer=model_nf4,
    torch_dtype=torch.bfloat16,
).to(device_img2img)
img2img_pipe = accelerator.prepare(img2img_pipe)
img2img_pipe.enable_model_cpu_offload()
img2img_pipe.safety_checker = None  # 필요에 따라 safety_checker 비활성화

## load inpaint2img model
device_inpaint2img = torch.device("cuda:3")
model_nf4 = SD3Transformer2DModel.from_pretrained(
    model_id,
    subfolder="transformer",
    quantization_config=nf4_config,
    torch_dtype=torch.bfloat16
).to(device_inpaint2img)

inpaint2img_pipe = StableDiffusion3InpaintPipeline.from_pretrained(
    model_id,
    transformer=model_nf4,
    torch_dtype=torch.bfloat16,
).to(device_inpaint2img)
inpaint2img_pipe = accelerator.prepare(inpaint2img_pipe)
inpaint2img_pipe.enable_model_cpu_offload()
inpaint2img_pipe.safety_checker = None  # 필요에 따라 safety_checker 비활성화

im = { 
  "background": "https://people.sc.fsu.edu/~jburkardt/data/png/dragon.png",
  "layers" : [],  
  "composite": None
}

# 이미지 전처리 함수: 흰 배경 유지, 투명 픽셀은 배경에 섞어 처리
def predict(im):
    comp = im["composite"]
    # NumPy 배열을 PIL 이미지로 변환 후 512x512로 리사이즈
    pil_img = Image.fromarray(comp.astype('uint8')).resize((512, 512))
    # RGBA 변환 (알파 채널 확보)
    pil_img = pil_img.convert("RGBA")
    
    # 흰색 배경 생성 후, 알파 채널을 이용해 원본 이미지 복사
    bg = Image.new("RGBA", pil_img.size, (255, 255, 255, 255))
    bg.paste(pil_img, (0, 0), pil_img)
    final_img = bg.convert("RGB")  # 최종 이미지는 RGB로 변환
    
    return np.array(final_img)

def mask(im):
    comp = im["background"]
    mask_image = im['layers'][0]
    # NumPy 배열을 PIL 이미지로 변환 후 512x512로 리사이즈
    # pil_img = Image.fromarray(comp.astype('uint8')).resize((512, 512))
    # # RGBA 변환 (알파 채널 확보)
    # pil_img = pil_img.convert("RGBA")
    
    # # 흰색 배경 생성 후, 알파 채널을 이용해 원본 이미지 복사
    # bg = Image.new("RGBA", pil_img.size, (255, 255, 255, 255))
    # bg.paste(pil_img, (0, 0), pil_img)
    # final_img = bg.convert("RGB")  # 최종 이미지는 RGB로 변환
    
    return mask_image

# 스케치 함수: composite 이미지를 받아 ControlNet을 통해 이미지 생성
def sketch(composite, prompt, negative_prompt, seed, num_steps):
    print("Sketching")
    if composite is None:
        print("composite is None")
        composite = np.full((512, 512, 3), 255, dtype=np.uint8)
    
    # 시드 설정
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    
    # composite 이미지를 흑백으로 변환
    composite_image = Image.fromarray(composite.astype(np.uint8)).convert("L")
    
    # 이미지 생성 (입력 이미지를 임계값 처리)
    images = pipe(
        prompt, 
        composite_image.convert("RGB").point(lambda p: 255 if p > 128 else 0), 
        negative_prompt=negative_prompt, 
        num_inference_steps=num_steps, 
        generator=generator, 
        controlnet_conditioning_scale=1.0
    ).images
    
    return images[0]

# run_sketching 함수: 스케치 이미지 여러 개 생성 및 shadow_draw 옵션 처리
def run_sketching(prompt, composite, sketch_states, shadow_draw):
    to_return = []
    curr_sketch = composite
    
    # num_images 개의 이미지를 생성
    for k in range(num_images):
        seed = sketch_states[k][1]
        if seed is None:
            seed = np.random.randint(1000)
            sketch_states[k][1] = seed
        new_image = sketch(curr_sketch, prompt, negative_prompt, seed=seed, num_steps=20)
        to_return.append(new_image)
    
    # shadow_draw 옵션이 True인 경우, HED를 사용해 평균 선 감지 처리
    if shadow_draw:
        hed_images = [hed(image, scribble=False) for image in to_return]
        avg_hed = np.mean([np.array(image) for image in hed_images], axis=0)
        curr_sketch_norm = np.array(curr_sketch).astype(float) / 255.
        curr_sketch = Image.fromarray(
            np.uint8((1.0 * (1. - (avg_hed / 255.))) * 255.)
        )
    else:
        curr_sketch = None
    
    # 결과 이미지 리스트에 composite 이미지 두 번과 sketch_states 추가
    return to_return + [curr_sketch, curr_sketch, sketch_states]

# 선택한 이미지를 반환하는 함수 (selected_image_display 업데이트용)
def select_image(image):
    return image
    
# image-to-image 생성 함수
def generate_img2img(selected_img, img2img_prompt, img2img_strength, img2img_guidance, img2img_negative_prompt):
    if selected_img is None:
        return None
    # PIL 이미지 변환 (이미 numpy array라고 가정)
    init_image = Image.fromarray(selected_img.astype('uint8')).resize((512,512))
    # 파이프라인 실행
    result = img2img_pipe(
        prompt=img2img_prompt,
        image=init_image,
        strength=img2img_strength,
        guidance_scale=img2img_guidance,
        negative_prompt=img2img_negative_prompt,
        num_inference_steps=1  # 필요에 따라 조정
    )
    return result.images[0]

# reset 함수: sketch_states 초기화
def reset(sketch_states):
    for k in range(num_images):
        sketch_states[k] = [None, None]
    return None, None, sketch_states

# Gradio 인터페이스 구성
with gr.Blocks() as demo:
    # 초기 sketch_states 설정
    start_state = [[None, None] for _ in range(num_images)]
    sketch_states = gr.State(start_state)
    checkbox_state = gr.State(True)
    white_brush = gr.Brush(default_color='#FFFFFF', colors=['#FFFFFF'], color_mode='fixed')

    with gr.Row():
        # 왼쪽 영역: 스케치 및 composite 미리보기 영역
        with gr.Column(scale=1):
            with gr.Tabs():
                with gr.TabItem("Sketch"):
                    i = gr.Sketchpad(brush=gr.Brush(default_size=3))
                with gr.TabItem("Suggested Lines"):
                    i_sketch = gr.Image()
            prompt_box = gr.Textbox(label="Prompt")
            with gr.Row():
                btn = gr.Button("Render")
                checkbox = gr.Checkbox(label="Generated suggested lines", value=True)
                btn2 = gr.Button("Reset")
            i_prev = gr.Image(label="Composite Preview", interactive=False)
        
        # 오른쪽 영역: 생성된 이미지 및 Select 버튼 영역
        with gr.Column(scale=1):
            out_imgs = []
            select_btns = []
            for idx in range(num_images):
                with gr.Column():
                    out_img = gr.Image(label=f"Generated Image {idx+1}", interactive=False)
                    select_btn = gr.Button("Select", variant="secondary")
                out_imgs.append(out_img)
                select_btns.append(select_btn)
        
        # 하단 영역: Image-to-Image 작업용 선택 이미지 및 추가 입력 컴포넌트
        with gr.Column():
            selected_image_display = gr.Image(label="Selected Image")
            img2img_prompt = gr.Textbox(label="Image-to-Image Prompt")
            img2img_negative_prompt = gr.Textbox(label="Negative Prompt")
            img2img_strength = gr.Slider(minimum=0, maximum=1, value=0.75, label="Strength (increase to ignore input image)")
            img2img_guidance = gr.Slider(minimum=1, maximum=10, value=7.5, label="Guidance Scale (increase to apply text prompt)")
            img2img_btn = gr.Button("Generate Image-to-Image")
            img2img_output = gr.Image(type='pil', label="Image-to-Image Result")
            img2img_select_btn = gr.Button("Select for Inpainting", variant="secondary")

        with gr.Column():
            inpaint2img = gr.ImageEditor(label='Inpaint', interactive=True)
            inpaint2img_prompt = gr.Textbox(label='prompt')
            inpaint2img_negative_prompt = gr.Textbox(label='negative prompt')
            inpaint2img_strength = gr.Slider(minimum=0, maximum=1, value=1.0, label="strength (increase inpainting strength)")
            inpaint2img_guidance = gr.Slider(minimum=1, maximum=10, value=7.5, label="guidance scale (increase to apply text prompt)")
            inpaint2img_outputs = gr.Image(type="pil", interactive=False)
            # gr.Interface(lambda x:x, gr.ImageEditor(), gr.ImageEditor(), examples=[im])
    # 이미지 편집이 변경될 때 predict 함수를 실행하여 composite 미리보기(i_prev) 갱신
    i.change(predict, inputs=i, outputs=i_prev, show_progress="hidden")
    
    # Render 버튼 클릭 시 run_sketching 실행
    # run_sketching의 출력 순서는:
    #  - num_images개의 생성된 이미지 (out_imgs)
    #  - i_sketch, i_prev, sketch_states (뒤 3개 출력)
    btn.click(
        run_sketching, 
        inputs=[prompt_box, i_prev, sketch_states, checkbox_state], 
        outputs=out_imgs + [i_sketch, i_prev, sketch_states]
    )
    
    # 각 Select 버튼 클릭 시, 해당 이미지를 selected_image_display로 업데이트
    for out_img, select_btn in zip(out_imgs, select_btns):
        select_btn.click(select_image, inputs=out_img, outputs=selected_image_display)
    
    # 이미지-투-이미지 버튼 클릭 시 generate_img2img 실행하여 결과를 img2img_output에 표시
    img2img_btn.click(
        generate_img2img,
        inputs=[selected_image_display, img2img_prompt, img2img_strength, img2img_guidance, img2img_negative_prompt],
        outputs=img2img_output
    )

    # img2img_output의 이미지를 inpaint2img의 입력 이미지로 전달하기 위해 select 버튼 클릭 이벤트 등록
    img2img_select_btn.click(
        select_image,
        inputs=img2img_output,
        outputs=inpaint2img
    )
    # inpaint2img.change(mask, inputs=inpaint2img, outputs=inpaint2img_outputs, show_progress="hidden" )
    # Reset 버튼 클릭 시 reset 함수 실행
    btn2.click(reset, inputs=sketch_states, outputs=[i, i_prev, sketch_states])
    
    # 체크박스 상태 변경 시 checkbox_state 업데이트
    checkbox.change(lambda x: x, inputs=[checkbox], outputs=[checkbox_state])

demo.launch(share=True)
