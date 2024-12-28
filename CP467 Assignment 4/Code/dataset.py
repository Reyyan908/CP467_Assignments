import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import copy

# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Dataset Preparation

# custom dataset for handwritten digits
class CustomMNISTDataset(Dataset):
    def __init__(self, image_folder, transform=None):
        self.image_folder = image_folder
        self.transform = transform
        self.images = []
        self.labels = []

        for img_name in os.listdir(image_folder): #go through all files in the image folder
            if img_name.endswith('.jpg'):
                img_path = os.path.join(image_folder, img_name) 
                self.images.append(img_path)
                label = int(img_name.split('_')[1].split('.')[0])  # extract digit from filename
                self.labels.append(label)

    def __len__(self): 
        return len(self.images) #number of images in dataset

    def __getitem__(self, idx):
        image = Image.open(self.images[idx])
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # average normalization parameters for MNIST
])

# dataset paths
root_folder = os.getcwd() #to create universal path
train_folder = os.path.join(root_folder, 'Training Set')  # contains 80 images
val_folder = os.path.join(root_folder, 'Validation Set')  # contains 20 images
test_folder = os.path.join(root_folder, 'Test Set')       # contains 20 images

# create datasets and dataloaders
train_dataset = CustomMNISTDataset(train_folder, transform=transform)
val_dataset = CustomMNISTDataset(val_folder, transform=transform)
test_dataset = CustomMNISTDataset(test_folder, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False, num_workers=0)

# Model Setup 

# create ResNet18 model adjusted for MNIST (28x28 grayscale images)
def create_resnet_model(pretrained=True, num_classes=10):
    model = models.resnet18(pretrained=pretrained)  # Load pre-trained weights
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False) #adjust for 1-channel images 
    
    # copy weights from original model and sum across channels
    if pretrained:
        pretrained_weights = models.resnet18(pretrained=True).conv1.weight
        model.conv1.weight.data = pretrained_weights.sum(dim=1, keepdim=True)
    
    # adjust fully connected layer to output num_classes
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

# function for fine-tuning only the last layer
def setup_model_1(model):
    for param in model.parameters(): # freeze all layers
        param.requires_grad = False
    for param in model.fc.parameters(): # unfreeze only the last fully connected layer
        param.requires_grad = True
    return model

# function for fine-tuning the last two layers
def setup_model_2(model):
    for param in model.parameters(): # freeze all layers
        param.requires_grad = False
    for name, param in model.named_parameters(): # unfreeze the last layer and layer4 (last residual block)
        if 'layer4' in name or 'fc' in name:
            param.requires_grad = True
    return model

# Training and Fine-tuning

def train_model(model, train_loader, criterion, optimizer, device):
    model.train()  # Set the model to training mode
    running_loss = 0.0
    correct = 0
    total = 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        
        # calculate training accuracy
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    return running_loss / len(train_loader), accuracy

def validate_model(model, val_loader, device):
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    return accuracy

def test_model(model, test_loader, device):
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    return accuracy


if __name__ == "__main__":
    # create pre-trained ResNet model
    resnet_pretrained = create_resnet_model(pretrained=True)
    resnet_pretrained.to(device)
    print("Loaded pre-trained ResNet model.")

    # evaluate the pre-trained model on MNIST test set
    mnist_test_dataset = datasets.MNIST(root='mnist_data', train=False, download=True, transform=transform)
    mnist_test_loader = DataLoader(mnist_test_dataset, batch_size=64, shuffle=False, num_workers=0)
    mnist_test_accuracy = test_model(resnet_pretrained, mnist_test_loader, device)
    print(f"Pre-trained ResNet Model on MNIST Test Set Accuracy: {mnist_test_accuracy:.2f}%")

    # evaluate the pre-trained model on your custom test set
    custom_test_accuracy = test_model(resnet_pretrained, test_loader, device)
    print(f"Pre-trained ResNet Model on Custom Test Set Accuracy: {custom_test_accuracy:.2f}%")

    # Fine-tuning Model 1: fine-tunes only the last layer
    print("\nTraining Model 1: Fine-tuning only the last layer.")
    model_1 = create_resnet_model(pretrained=True)
    model_1 = setup_model_1(model_1)
    model_1.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer_1 = optim.Adam(filter(lambda p: p.requires_grad, model_1.parameters()), lr=0.001) # hyperparameters (adjusted to prevent overfitting)
    num_epochs = 20
    best_model_wts = copy.deepcopy(model_1.state_dict())
    best_val_accuracy = 0.0

    train_accuracies = []
    val_accuracies = []

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        # train the model
        train_loss, train_accuracy = train_model(model_1, train_loader, criterion, optimizer_1, device)
        train_accuracies.append(train_accuracy)
        print(f"Training Loss: {train_loss:.4f}, Training Accuracy: {train_accuracy:.2f}%")

        # validate the model
        val_accuracy = validate_model(model_1, val_loader, device)
        val_accuracies.append(val_accuracy)
        print(f"Validation Accuracy: {val_accuracy:.2f}%")

        # deep copy the model if it has better validation accuracy
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_model_wts = copy.deepcopy(model_1.state_dict())

    # load best model weights
    model_1.load_state_dict(best_model_wts)

    # evaluate Model 1 on test set
    model_1_test_accuracy = test_model(model_1, test_loader, device)
    print(f"Model 1 (Fine-tune last layer) Test Accuracy: {model_1_test_accuracy:.2f}%")

    # Fine-tuning Model 2: fine-tunes two or more layers
    print("\nTraining Model 2: Fine-tuning last two layers.")
    model_2 = create_resnet_model(pretrained=True)
    model_2 = setup_model_2(model_2)
    model_2.to(device)
 
    optimizer_2 = optim.Adam(filter(lambda p: p.requires_grad, model_2.parameters()), lr=0.0001) # hyperparameters (adjusted to prevent overfitting)
    best_model_wts = copy.deepcopy(model_2.state_dict())
    best_val_accuracy = 0.0

    train_accuracies_2 = []
    val_accuracies_2 = []

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        # train the model
        train_loss, train_accuracy = train_model(model_2, train_loader, criterion, optimizer_2, device)
        train_accuracies_2.append(train_accuracy)
        print(f"Training Loss: {train_loss:.4f}, Training Accuracy: {train_accuracy:.2f}%")

        # validate the model
        val_accuracy = validate_model(model_2, val_loader, device)
        val_accuracies_2.append(val_accuracy)
        print(f"Validation Accuracy: {val_accuracy:.2f}%")

        # deep copy the model if it has better validation accuracy
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_model_wts = copy.deepcopy(model_2.state_dict())

    # load best model weights
    model_2.load_state_dict(best_model_wts)

    # evaluate Model 2 on test set
    model_2_test_accuracy = test_model(model_2, test_loader, device)
    print(f"Model 2 (Fine-tune last two layers) Test Accuracy: {model_2_test_accuracy:.2f}%")

    # Results:
    print("\n--- Final Results ---")
    print(f"1. Pre-trained ResNet Model on MNIST Test Set Accuracy: {mnist_test_accuracy:.2f}%")
    print(f"2. Pre-trained ResNet Model on Custom Test Set Accuracy: {custom_test_accuracy:.2f}%")
    print(f"3. Model 1 (Fine-tune last layer)")
    print(f"   Training Accuracy: {train_accuracies[-1]:.2f}%")
    print(f"   Validation Accuracy: {val_accuracies[-1]:.2f}%")
    print(f"   Test Accuracy: {model_1_test_accuracy:.2f}%")
    print(f"4. Model 2 (Fine-tune last two layers) Training Accuracy")
    print(f"   Training Accuracy: {train_accuracies_2[-1]:.2f}%")
    print(f"   Validation Accuracy: {val_accuracies_2[-1]:.2f}%")
    print(f"   Test Accuracy: {model_2_test_accuracy:.2f}%")
