# predict.py
import torch
from torchvision import transforms, models
from PIL import Image
import torch.nn as nn

IMG_SIZE = 224
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# same classes in alphabetical order as your folders
classes = ['glioma', 'meningioma', 'notumour', 'pituitary']

# Load model
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(classes))
model.load_state_dict(torch.load("brain_tumor_model.pth", map_location=DEVICE))
model.to(DEVICE)
model.eval()

# Preprocess image
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Enter your test image path here
img_path = "dataset\Testing\meningioma\Te-me_0048.jpg" # change this
img = Image.open(img_path).convert("RGB")
x = transform(img).unsqueeze(0).to(DEVICE)

with torch.no_grad():
    outputs = model(x)
    _, pred = torch.max(outputs, 1)
    prediction = classes[pred.item()]

print(f"🧠 Predicted Tumor Type: {prediction}")
