from flask import Flask, render_template, request
from torchvision import transforms, models
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import os

# ================= APP SETUP =================
app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
HEATMAP_FOLDER = "static/heatmaps"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(HEATMAP_FOLDER, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================= CLASSES =================
classes = ['glioma', 'meningioma', 'notumor', 'pituitary']

# ================= MEDICAL DATABASE =================
tumor_info = {
    "glioma": {
        "about": "Glioma is a tumor arising from glial cells. It can grow aggressively and press on brain tissue, causing headaches, seizures, personality changes, and cognitive problems. Early diagnosis is crucial as higher-grade gliomas may spread rapidly and require intensive treatment and long-term monitoring.",
        "severity": "High",
        "risk": "High Risk",
        "diet_do": [
            "Leafy vegetables and fruits",
            "Omega-3 rich foods",
            "Whole grains",
            "Lean proteins"
        ],
        "diet_dont": [
            "Processed foods",
            "Sugary items",
            "Alcohol",
            "Smoking"
        ],
        "treatment": "Surgery, Chemotherapy, Radiotherapy",
        "specialist": "Neurosurgeon / Neuro-oncologist"
    },
    "meningioma": {
        "about": "Meningioma is usually a slow-growing tumor that develops from the brain’s protective membranes. It can compress nearby brain areas, leading to headaches, vision issues, seizures, or memory problems. Though mostly non-cancerous, medical monitoring or treatment is often required.",
        "severity": "Moderate",
        "risk": "Medium Risk",
        "diet_do": [
            "Fresh fruits and vegetables",
            "Whole grains",
            "Fish and nuts",
            "Adequate hydration"
        ],
        "diet_dont": [
            "Fried food",
            "Excess salt",
            "Refined sugar",
            "Alcohol"
        ],
        "treatment": "Observation, Surgery, Radiation",
        "specialist": "Neurosurgeon"
    },
    "pituitary": {
        "about": "Pituitary tumors affect hormone regulation and may cause vision problems, fatigue, weight changes, and hormonal imbalance. Although many are benign, they can significantly impact body functions and often require medication, surgery, or radiation therapy.",
        "severity": "Moderate",
        "risk": "Medium Risk",
        "diet_do": [
            "High-fiber foods",
            "Protein-rich meals",
            "Fresh vegetables",
            "Plenty of water"
        ],
        "diet_dont": [
            "Sugary foods",
            "Processed snacks",
            "Excess caffeine",
            "Alcohol"
        ],
        "treatment": "Medication, Surgery, Radiation",
        "specialist": "Endocrinologist / Neurosurgeon"
    },
    "notumor": {
        "about": "No tumor has been detected. The brain MRI appears normal with no abnormal growth or compression. Maintaining a healthy lifestyle and regular medical checkups is advised.",
        "severity": "None",
        "risk": "No Risk",
        "diet_do": [
            "Balanced diet",
            "Regular exercise",
            "Adequate sleep"
        ],
        "diet_dont": [
            "Smoking",
            "Excess alcohol"
        ],
        "treatment": "No treatment required",
        "specialist": "-"
    }
}

# ================= MODEL =================
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(classes))
model.load_state_dict(torch.load("brain_tumor_model.pth", map_location=DEVICE))
model.to(DEVICE)
model.eval()

# ================= TRANSFORM =================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

# ================= GRAD-CAM =================
class GradCAM:
    def __init__(self, model):
        self.model = model
        self.gradients = None
        self.activations = None

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        def forward_hook(module, input, output):
            self.activations = output

        model.layer4.register_forward_hook(forward_hook)
        model.layer4.register_backward_hook(backward_hook)

    def generate(self, x, class_idx):
        output = self.model(x)
        self.model.zero_grad()
        output[0, class_idx].backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1).squeeze()
        cam = cam.detach().cpu().numpy()

        cam = np.maximum(cam, 0)
        cam = cam / (cam.max() + 1e-8)
        return cam

gradcam = GradCAM(model)

# ================= PREDICTION =================
def predict_tumor(image_path):
    img = Image.open(image_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(x)
        probs = F.softmax(outputs, dim=1)
        confidence, idx = torch.max(probs, 1)

    return classes[idx.item()], round(confidence.item() * 100, 2), x, idx.item()

# ================= HEATMAP =================
def generate_heatmap(image_path, tensor, class_idx):
    cam = gradcam.generate(tensor, class_idx)

    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))

    cam = cv2.resize(cam, (224, 224))
    cam = np.uint8(255 * cam)

    heatmap = cv2.applyColorMap(cam, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)

    heatmap_path = os.path.join(HEATMAP_FOLDER, os.path.basename(image_path))
    cv2.imwrite(heatmap_path, overlay)
    return heatmap_path

# ================= ROUTES =================
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def upload_image():
    if "file" not in request.files:
        return "No file uploaded", 400

    file = request.files["file"]
    if file.filename == "":
        return "No image selected", 400

    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    tumor, confidence, tensor, class_idx = predict_tumor(filepath)
    heatmap_path = generate_heatmap(filepath, tensor, class_idx)
    info = tumor_info[tumor]

    return render_template(
        "result.html",
        image_path=filepath,
        heatmap_path=heatmap_path,
        result=tumor,
        confidence=confidence,
        info=info,
        disclaimer="This system is for educational and research purposes only and not for medical diagnosis."
    )

# ================= MAIN =================
if __name__ == "__main__":
    app.run(debug=True)
