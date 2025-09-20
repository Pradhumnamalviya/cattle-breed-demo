import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import gradio as gr

# ------------------------------
# 1. Load your trained model
# ------------------------------
# If the .pt file contains the full model (not just state_dict):
model = torch.load("bovinebreedclassifier.pt", map_location=torch.device("cpu"), weights_only=False)

# If it's only state_dict, then uncomment this and define your model architecture
# class BovineBreedClassifier(nn.Module):
#     def __init__(self, num_classes=10):  # adjust num_classes
#         super(BovineBreedClassifier, self).__init__()
#         # Example architecture (replace with your actual one)
#         self.model = torchvision.models.resnet18(pretrained=False)
#         self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
#
#     def forward(self, x):
#         return self.model(x)
#
# model = BovineBreedClassifier(num_classes=YOUR_CLASSES)
# model.load_state_dict(torch.load("bovinebreedclassifier.pt", map_location="cpu"))

model.eval()

# ------------------------------
# 2. Define transforms (same as training)
# ------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),   # Adjust to your input size
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Standard ImageNet normalization
])

# ------------------------------
# 3. Class labels (update with your breeds)
# ------------------------------
class_names = [
    'Alambadi', 'Amritmahal', 'Ayrshire', 'Banni', 'Bargur', 'Bhadawari',
    'Brown_Swiss', 'Dangi', 'Deoni', 'Gir', 'Guernsey', 'Hallikar',
    'Hariana', 'Holstein_Friesian', 'Jaffrabadi', 'Jersey', 'Kangayam',
    'Kankrej', 'Kasargod', 'Kenkatha', 'Kherigarh', 'Khillari',
    'Krishna_Valley', 'Malnad_gidda', 'Mehsana', 'Murrah', 'Nagori',
    'Nagpuri', 'Nili_Ravi', 'Nimari', 'Ongole', 'Pulikulam', 'Rathi',
    'Red_Dane', 'Red_Sindhi', 'Sahiwal', 'Surti', 'Tharparkar', 'Toda',
    'Umblachery', "Vechur"
]

# ------------------------------
# 4. Prediction function
# ------------------------------
def predict(image):
    image = Image.fromarray(image).convert("RGB")
    img_t = transform(image).unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        outputs = model(img_t)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        top_prob, top_class = torch.max(probabilities, dim=0)

    return {class_names[top_class.item()]: float(top_prob)}

# ------------------------------
# 5. Gradio Interface
# ------------------------------
demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="numpy"),
    outputs=gr.Label(num_top_classes=3),
    title="🐄 Cattle & Buffalo Breed Classifier",
    description="Upload an image of a cow or buffalo to identify its breed."
)

# ------------------------------
# 6. Launch
# ------------------------------
if __name__ == "__main__":
    demo.launch()
