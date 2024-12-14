from flask import Flask, request, render_template_string, send_from_directory
import os
from PIL import Image
import torch
from RealESRGAN import RealESRGAN

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = "uploads"
RESULT_FOLDER = "results"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# Device setup for RealESRGAN
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = RealESRGAN(device, scale=4)
model.load_weights('weights/RealESRGAN_x4.pth', download=True)

# HTML Template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Image Enhancer</title>
    <style>
        body { font-family: Arial, sans-serif; background-color: #f4f4f9; margin: 0; padding: 20px; }
        .container { max-width: 600px; margin: auto; padding: 20px; background: #fff; border-radius: 8px; box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1); }
        h1 { text-align: center; color: #333; }
        form { display: flex; flex-direction: column; gap: 10px; }
        input[type="file"] { padding: 10px; }
        button { padding: 10px; background: #4CAF50; color: #fff; border: none; cursor: pointer; }
        button:hover { background: #45a049; }
        .images { display: flex; flex-direction: column; align-items: center; gap: 20px; margin-top: 20px; }
        img { max-width: 100%; border: 1px solid #ddd; border-radius: 8px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>AI Image Enhancer</h1>
        <form method="POST" enctype="multipart/form-data">
            <input type="file" name="image" accept="image/*" required>
            <button type="submit">Enhance Image</button>
        </form>
        {% if input_image and output_image %}
        <div class="images">
            <div>
                <h3>Input Image</h3>
                <img src="{{ input_image }}" alt="Input Image">
            </div>
            <div>
                <h3>Enhanced Image</h3>
                <img src="{{ output_image }}" alt="Enhanced Image">
            </div>
        </div>
        {% endif %}
    </div>
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def enhance_image():
    input_image_path = None
    output_image_path = None

    if request.method == "POST":
        # Save uploaded image
        uploaded_file = request.files["image"]
        if uploaded_file:
            input_image_path = os.path.join(UPLOAD_FOLDER, uploaded_file.filename)
            uploaded_file.save(input_image_path)

            # Open and enhance the image
            image = Image.open(input_image_path).convert("RGB")
            enhanced_image = model.predict(image)

            # Save enhanced image
            output_image_path = os.path.join(RESULT_FOLDER, f"enhanced_{uploaded_file.filename}")
            enhanced_image.save(output_image_path)

    # Serve the page with images if processed
    return render_template_string(
        HTML_TEMPLATE,
        input_image=f"/uploads/{os.path.basename(input_image_path)}" if input_image_path else None,
        output_image=f"/results/{os.path.basename(output_image_path)}" if output_image_path else None
    )

@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route("/results/<filename>")
def result_file(filename):
    return send_from_directory(RESULT_FOLDER, filename)

if __name__ == "__main__":
    app.run(debug=True)
