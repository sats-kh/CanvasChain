import gradio as gr
import torch
import numpy as np
from PIL import Image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, EulerAncestralDiscreteScheduler
from diffusers.utils import load_image
from controlnet_aux import HEDdetector

# Model and constant initialization (동일)
negative_prompt = ""
device = torch.device('cuda')
num_images = 3

controlnet = ControlNetModel.from_pretrained(
    "vsanimator/sketch-a-sketch", 
    torch_dtype=torch.float16
).to(device)

pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", 
    controlnet=controlnet, 
    torch_dtype=torch.float16
).to(device)
pipe.safety_checker = None
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
hed = HEDdetector.from_pretrained('lllyasviel/Annotators')  # ControlNet

######################################
# Functions
######################################

def sketch(curr_sketch, prev_sketch, prompt, negative_prompt, seed, num_steps=20):
    """주어진 스케치를 기반으로 이미지를 생성"""
    print("Sketching")
    if curr_sketch is None:
        curr_sketch = np.full((512, 512, 3), 255, dtype=np.uint8)
    if prev_sketch is None:
        prev_sketch = curr_sketch
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    
    # curr_sketch는 gr.ImageEditor에서 전달된 경우 dict형태이므로 "composite" 키를 사용
    if isinstance(curr_sketch, dict):
        curr_sketch = curr_sketch.get("composite")
    
    curr_sketch_image = Image.fromarray(curr_sketch.astype(np.uint8)).convert("L")
    
    images = pipe(
        prompt,
        curr_sketch_image.convert("RGB").point(lambda p: 256 if p > 128 else 0),
        negative_prompt=negative_prompt,
        num_inference_steps=num_steps,
        generator=generator,
        controlnet_conditioning_scale=1.0
    ).images
    
    return images[0]
def run_sketching(prompt, curr_sketch, prev_sketch, sketch_states, shadow_draw):
    """여러 이미지를 생성하고 스케치 상태를 관리 (gr.ImageEditor 사용)"""
    # gr.ImageEditor로부터 전달된 값은 dict 형태일 수 있으므로, composite 이미지를 추출
    if curr_sketch is not None and isinstance(curr_sketch, dict):
        curr_sketch_np = curr_sketch.get("composite")
    else:
        curr_sketch_np = curr_sketch

    to_return = []
    for k in range(num_images):
        seed = sketch_states[k][1]
        if seed is None:
            seed = np.random.randint(1000)
            sketch_states[k][1] = seed
        new_image = sketch(
            curr_sketch_np, prev_sketch, prompt,
            negative_prompt, seed=seed, num_steps=20
        )
        to_return.append(new_image)

    if curr_sketch_np is None:
        curr_sketch_np = np.full((512, 512, 3), 255, dtype=np.uint8)
    prev_sketch = curr_sketch_np

    if shadow_draw:
        hed_images = []
        for image in to_return:
            # hed()로 반환된 이미지를 PIL 이미지로 변환, RGB로 변환 후 (512,512)로 리사이즈
            hed_image = hed(image, scribble=False)
            pil_hed = Image.fromarray(np.uint8(hed_image)).convert("RGB").resize((512,512))
            hed_images.append(np.array(pil_hed).astype(float))
        # 여러 hed 이미지를 평균내어 하나의 이미지 생성
        avg_hed_resized = np.mean(hed_images, axis=0)
        # 혹시 avg_hed_resized의 크기가 맞지 않으면 강제로 (512,512,3)으로 변환
        if avg_hed_resized.shape[:2] != (512,512) or avg_hed_resized.shape[2] != 3:
            avg_hed_resized = np.array(
                Image.fromarray(np.uint8(avg_hed_resized)).convert("RGB").resize((512,512))
            ).astype(float)
        # 원본 스케치도 0~1 사이 값으로 변환 (여기서는 사용되지 않지만, 필요시 대비)
        curr_sketch_arr = np.array(curr_sketch_np).astype(float) / 255.
        # 아래 계산식은 원본 스케치를 무시하고 avg_hed를 기반으로 결과를 생성함 (필요에 따라 조정)
        curr_sketch_final = Image.fromarray(
            np.uint8( (1.0 - (avg_hed_resized / 255.)) * 255. )
        )
    else:
        curr_sketch_final = None

    return to_return + [curr_sketch_final, prev_sketch, sketch_states]
    
def reset(sketch_states):
    """스케치 상태 초기화"""
    for k in range(num_images):
        sketch_states[k] = [None, None]
    return None, None, sketch_states

######################################
# UI 구성 (Gradio 5.x)
######################################

with gr.Blocks() as demo:
    # 초기 상태 설정
    initial_sketch_state = [[None, None] for _ in range(num_images)]
    sketch_states = gr.State(initial_sketch_state)
    checkbox_state = gr.State(True)
    
    with gr.Row():
        with gr.Column(scale=1):
            # Tabs 사용: Sketch (입력)와 Suggested Lines (결과 미리보기)
            with gr.Tabs():
                with gr.TabItem("Sketch"):
                    i = gr.ImageEditor(
                        type="numpy",
                        height=600,
                        width=600
                    )
                with gr.TabItem("Suggested Lines"):
                    i_sketch = gr.Image(
                        height=600,
                        width=600
                    )
            prompt_box = gr.Textbox(label="Prompt")
            with gr.Row():
                btn = gr.Button("Render")
                checkbox = gr.Checkbox(label="Generated suggested lines", value=True)
                btn2 = gr.Button("Reset")
            i_prev = gr.Image(height=768, width=768)
        with gr.Column(scale=1):
            o_list = [gr.Image(height=512, width=512) for _ in range(num_images)]
    
    # 이벤트 연결
    btn.click(
        fn=run_sketching,
        inputs=[prompt_box, i, i_prev, sketch_states, checkbox_state],
        outputs=o_list + [i_sketch, i_prev, sketch_states]
    )
    
    btn2.click(
        fn=reset,
        inputs=sketch_states,
        outputs=[i, i_prev, sketch_states]
    )
    
    checkbox.change(
        fn=lambda x: x,
        inputs=[checkbox],
        outputs=[checkbox_state]
    )

demo.launch(share=True)