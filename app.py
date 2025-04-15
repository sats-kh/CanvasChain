import gradio as gr
from model_functions import (
    predict, run_sketching, generate_img2img,
    generate_inpainted_image, reset
)
import numpy as np

# 기본 설정값
num_images = 6

# --- Wrapper 함수 정의 (기존 코드는 그대로 유지) ---
def run_sketching_with_version(selected_model, prompt, negative_prompt, i_prev, sketch_states, checkbox_state, checkbox_remove_background, version_state):
    outputs = run_sketching(selected_model, prompt, negative_prompt, i_prev, sketch_states, checkbox_state, checkbox_remove_background)
    if i_prev is not None:
        version_state.append((i_prev, f"Sketch Composite\nPrompt: {prompt}\nNegative_prompt: {negative_prompt}"))
        return outputs + [version_state]
    return outputs + [version_state]

def generate_img2img_with_version(selected_image, prompt, strength, guidance, negative, version_state, checkbox_remove_background):
    result = generate_img2img(selected_image, prompt, strength, guidance, negative, checkbox_remove_background)
    if version_state is None:
        version_state = []
    version_state.append((result, f"Img2Img\nPrompt: {prompt}\nNegative_prompt: {negative}\nStrength: {strength}, Guidance: {guidance}"))
    return result, version_state

def generate_inpainted_image_with_version(inpaint_img, prompt, negative, strength, guidance, version_state, checkbox_remove_background):
    result = generate_inpainted_image(inpaint_img, prompt, negative, strength, guidance, checkbox_remove_background)
    if version_state is None:
        version_state = []
    version_state.append((result, f"Inpainting\nPrompt: {prompt}\nNegative_prompt: {negative}\nStrength: {strength}, Guidance: {guidance}"))
    return result, version_state

def select_image(image):
    if isinstance(image, (list, tuple)):
        return image[0]
    return image

def select_image_and_record_switch_with_prompts(image, version_state, prompt, negative_prompt):
    selected = select_image(image)
    if version_state is None:
        version_state = []
    version_state.append((selected, f"Sketch2Image\nPrompt: {prompt}\nNegative_prompt: {negative_prompt}"))
    return selected, version_state, prompt, negative_prompt, gr.update(selected=1)

def select_image_and_update_tab_with_prompts(image, prompt, negative_prompt):
    selected = select_image(image)
    return selected, prompt, negative_prompt, gr.update(selected=2)

# Function to handle gallery selection
def on_gallery_select(evt: gr.SelectData):
    return evt.index

# Function to pass the selected image to the img2img
def pass_selected_to_img2img(gallery_value, idx):
    if idx is not None and gallery_value is not None and len(gallery_value) > idx:
        image, _ = gallery_value[idx]  # 튜플에서 이미지 추출 (캡션 무시)
        return image, gr.update(selected=1)
    else:
        return None

# Function to pass the selected image to the inpaint2img
def pass_selected_to_inpaint2img(gallery_value, idx):
    if idx is not None and gallery_value is not None and len(gallery_value) > idx:
        image, _ = gallery_value[idx]  # 튜플에서 이미지 추출 (캡션 무시)
        return image, gr.update(selected=2)
    else:
        return None

css_snippet = """
/* 푸터 영역을 강제로 숨기는 CSS */
.gradio-container footer {
  display: none !important;
}
"""


# --- UI 구성 ---
with gr.Blocks(css=css_snippet) as demo:
    version_state = gr.State([])
    selected_gallery_index = gr.State(value=None)

    start_state = [[None, None] for _ in range(num_images)]
    sketch_states = gr.State(start_state)
    white_brush = gr.Brush(default_color='#FFFFFF', colors=['#FFFFFF'], color_mode='fixed')

    with gr.Row():
        with gr.Column(scale=1):
            with gr.Tabs() as sub_tabs:
                with gr.TabItem("Sketch", id=0):
                    with gr.Row(scale=2):
                        with gr.Column(scale=1):
                            i = gr.Sketchpad(brush=gr.Brush(default_size=3), canvas_size=(1024,1024))
                            model_dropdown = gr.Dropdown(
                                choices=["Ghibli-Diffusion", "Disney-Diffusion"],
                                label="Select Model for Sketch-to-Image",
                                value="Ghibli-Diffusion"
                            )
                            sketch_prompt = gr.Textbox(label="Prompt")
                            skectch_negative_prompt = gr.Textbox(label="Negative Prompt")
                            checkbox_for_suggested_lines = gr.Checkbox(label="Generated suggested lines", value=True)
                            checkbox_remove_background = gr.Checkbox(label="Remove Background", value=False)
                            with gr.Row():
                                btn = gr.Button("Render")
                                btn2 = gr.Button("Reset")
                            i_prev = gr.Image(label="Composite Preview", interactive=False)
                        with gr.Column(scale=1):
                            out_imgs = []
                            select_btns = []
                            for r in range(3):  # 3행
                                with gr.Row():
                                    for c in range(2):  # 2열
                                        idx = r * 2 + c
                                        if idx < num_images:
                                            with gr.Column():
                                                out_img = gr.Image(label=f"Generated Image {idx+1}", interactive=False)
                                                select_btn = gr.Button("Select", variant="secondary")
                                            out_imgs.append(out_img)
                                            select_btns.append(select_btn)
                with gr.TabItem("Suggested Lines", visible=False):
                    i_sketch = gr.Image()
                with gr.TabItem("Image-to-Image", id=1):
                    with gr.Row(scale=2):
                        with gr.Column():
                            selected_image_display = gr.Image(label="Selected Image")
                            img2img_prompt = gr.Textbox(label="Image-to-Image Prompt")
                            img2img_negative_prompt = gr.Textbox(label="Negative Prompt")
                            checkbox_remove_background_for_img2img = gr.Checkbox(label="Remove Background", value=False)
                            img2img_strength = gr.Slider(minimum=0, maximum=1, value=0.75, label="Strength")
                            img2img_guidance = gr.Slider(minimum=1, maximum=10, value=7.5, label="Guidance Scale")
                            img2img_btn = gr.Button("Generate Image-to-Image")
                        with gr.Column():
                            img2img_output = gr.Image(type='pil', label="Image-to-Image Result")
                            img2img_select_btn = gr.Button("Select for Inpainting", variant="secondary")
                with gr.TabItem("Inpaint-to-Image", id=2):
                    with gr.Row(scale=2):
                        with gr.Column():
                            inpaint2img = gr.ImageEditor(label='Inpaint', interactive=True, canvas_size=(1024,1024))
                            inpaint2img_prompt = gr.Textbox(label='prompt')
                            inpaint2img_negative_prompt = gr.Textbox(label='negative prompt')
                            checkbox_remove_background_for_inpaint2img = gr.Checkbox(label="Remove Background", value=False)
                            inpaint2img_strength = gr.Slider(minimum=0, maximum=1, value=1.0, label="Strength")
                            inpaint2img_guidance = gr.Slider(minimum=1, maximum=10, value=7.5, label="Guidance Scale")
                            inpaint2img_btn = gr.Button("Generate Inpainting")
                        with gr.Column():
                            inpaint2img_outputs = gr.Image(type="pil", interactive=False)
                            inpaint2img_retry_btn = gr.Button("Retry", variant="secondary")
                with gr.TabItem("Version", id=3):
                    version_gallery = gr.Gallery(
                        label="Version History",
                        columns=[5],
                        object_fit="contain",
                        height="auto",
                        interactive=True
                    )
                    with gr.Row(scale=2):
                        with gr.Column():
                            from_gallery_to_img2img = gr.Button("Select for Image2Image")
                        with gr.Column():
                            from_gallery_to_inpaint2img = gr.Button("Select for Inpaint2Image")

    # --- 이벤트/콜백 연결 ---
    i.change(predict, inputs=i, outputs=i_prev, show_progress="hidden")
    
    btn.click(
        run_sketching_with_version, 
        inputs=[model_dropdown, sketch_prompt, skectch_negative_prompt, i_prev, sketch_states, checkbox_for_suggested_lines, checkbox_remove_background, version_state], 
        outputs=out_imgs + [i_sketch, i_prev, sketch_states, version_state]
    )
    
    for out_img, select_btn in zip(out_imgs, select_btns):
        select_btn.click(
            select_image_and_record_switch_with_prompts,
            inputs=[out_img, version_state, sketch_prompt, skectch_negative_prompt],
            outputs=[selected_image_display, version_state, img2img_prompt, img2img_negative_prompt, sub_tabs]
        )
    
    img2img_btn.click(
        generate_img2img_with_version,
        inputs=[selected_image_display, img2img_prompt, img2img_strength, img2img_guidance, img2img_negative_prompt, version_state, checkbox_remove_background_for_img2img],
        outputs=[img2img_output, version_state]
    )
    
    img2img_select_btn.click(
        select_image_and_update_tab_with_prompts,
        inputs=[img2img_output, img2img_prompt, img2img_negative_prompt],
        outputs=[inpaint2img, inpaint2img_prompt, inpaint2img_negative_prompt, sub_tabs]
    )
    inpaint2img_retry_btn.click(
        select_image,
        inputs=inpaint2img_outputs,
        outputs=inpaint2img
    )

    inpaint2img_btn.click(
        generate_inpainted_image_with_version,
        inputs=[inpaint2img, inpaint2img_prompt, inpaint2img_negative_prompt, inpaint2img_strength, inpaint2img_guidance, version_state, checkbox_remove_background_for_inpaint2img], 
        outputs=[inpaint2img_outputs, version_state]
    )
    
    btn2.click(reset, inputs=sketch_states, outputs=[i, i_prev, sketch_states])
    
    version_state.change(lambda x: x, inputs=[version_state], outputs=[version_gallery])
    
    version_gallery.select(on_gallery_select, None, selected_gallery_index)
    from_gallery_to_img2img.click(pass_selected_to_img2img, inputs=[version_gallery,selected_gallery_index], outputs=[selected_image_display, sub_tabs])
    from_gallery_to_inpaint2img.click(pass_selected_to_inpaint2img, inputs=[version_gallery,selected_gallery_index], outputs=[inpaint2img, sub_tabs])

demo.launch(server_name="0.0.0.0", server_port=7860)