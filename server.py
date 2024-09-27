from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS
from diffusers import StableDiffusionPipeline
import torch
from PIL import Image
import base64
from io import BytesIO

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the Stable Diffusion model
model_id = "CompVis/stable-diffusion-v1-4"
pipe = StableDiffusionPipeline.from_pretrained(model_id)
pipe = pipe.to("cpu")  # Use CPU for processing

@app.route('/generate', methods=['POST'])
def generate():
    # Get the text prompt from the request
    data = request.get_json()
    prompt = data['prompt']

    # Generate the image
    image = pipe(prompt).images[0]

    # Convert the image to base64
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')

    # Return the image as base64
    return jsonify({'image': img_str})

if __name__ == '__main__':
    app.run(debug=True)
