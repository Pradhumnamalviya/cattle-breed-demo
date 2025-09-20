import torch

# Load the trained model
model = torch.load("bovinebreedclassifier.pt", map_location=torch.device("cpu"), weights_only=False)

print("✅ Model loaded successfully")
print(model)



