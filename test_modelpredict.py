import timm
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def test_mini_imagenet(data_path, batch_size=32):
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define transform (resize to 224x224 for ViT)
    
    transform = transforms.Compose([
        transforms.Resize(224),           # Resize ngắn nhất về 224
        transforms.CenterCrop(224),       # Crop trung tâm thành 224x224
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


    # Load validation dataset
    val_dataset = datasets.ImageFolder(root=f"{data_path}/validation", transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Load pre-trained model
    model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=100)
    model = model.to(device)
    model.eval()  # Set model to evaluation mode

    # Initialize metrics
    correct = 0
    total = 0

    with torch.no_grad():  # No gradients needed for testing
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total
    print(f"Validation Accuracy: {accuracy:.2%}")

# Test the function
data_path = "/mnt/data/mini-imagenet"  # Path to dataset
test_mini_imagenet(data_path)

