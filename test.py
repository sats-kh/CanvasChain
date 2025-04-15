import gradio as gr
from PIL import Image
import requests
import io

# Utility function to load an image from a URL
def load_image(url):
    response = requests.get(url)
    return Image.open(io.BytesIO(response.content))

# Sample image URLs
sample_urls = [
    "http://www.marketingtool.online/en/face-generator/img/faces/avatar-1151ce9f4b2043de0d2e3b7826127998.jpg",
    "http://www.marketingtool.online/en/face-generator/img/faces/avatar-116b5e92936b766b7fdfc242649337f7.jpg",
    "http://www.marketingtool.online/en/face-generator/img/faces/avatar-1163530ca19b5cebe1b002b8ec67b6fc.jpg",
    "http://www.marketingtool.online/en/face-generator/img/faces/avatar-1116395d6e6a6581eef8b8038f4c8e55.jpg",
    "http://www.marketingtool.online/en/face-generator/img/faces/avatar-11319be65db395d0e8e6855d18ddcef0.jpg",
]

# Load images from URLs
sample_images = [load_image(url) for url in sample_urls]

# Function to handle gallery selection
def on_gallery_select(evt: gr.SelectData):
    return evt.index

# Function to pass the selected image to the editor
def pass_selected_to_editor(gallery_value, idx):
    if idx is not None and gallery_value is not None and len(gallery_value) > idx:
        image, _ = gallery_value[idx]  # 튜플에서 이미지 추출 (캡션 무시)
        return image
    else:
        return None

with gr.Blocks() as demo:
    gr.Markdown("## Sample Gallery to Image Editor Example")
    
    # Gallery component displaying sample images
    gallery = gr.Gallery(
        value=sample_images,
        label="Sample Gallery"
    )
    
    # State to hold the selected index
    selected_index = gr.State(value=None)
    
    # Set up select event for the gallery
    gallery.select(on_gallery_select, None, selected_index)
    
    # Image component (editor)
    image_editor = gr.Image(
        label="Image Editor"
    )
    
    # Button that passes the selected image to the editor
    btn = gr.Button("Edit Selected Image")
    
    # Set up button click to pass the selected image
    btn.click(pass_selected_to_editor, inputs=[gallery, selected_index], outputs=image_editor)

demo.launch(share=True)