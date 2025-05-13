# Import Libraries
import os 
from PIL import Image 
import torch
import torch.nn as nn 
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from sklearn.metrics import accuracy_score

# Enable GPU for computing if exists
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define main path 
main_path = "D://study//Machine Learning//Projects//Pneumonia_Classifier//data//chest_xray//chest_xray"

# Define dataset class extended from DataSet
class X_Ray_Dataset(Dataset):
    
    # Constructor
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir 
        self.transform = transform 
        self.image_paths = [] 
        self.labels = []

        # Valid image extensions
        valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']

        for label in ['NORMAL','PNEUMONIA']:
            class_dir = os.path.join(root_dir, label)
            for img_name in os.listdir(class_dir):
                # Check if file is a valid image file
                file_extension = os.path.splitext(img_name)[1].lower()
                if file_extension in valid_extensions:
                    self.image_paths.append(os.path.join(class_dir, img_name))
                    # Fixed typo in 'NORMAL' condition
                    self.labels.append(0 if label=='NORMAL' else 1)
    
    # Dataset length
    def __len__(self):
        return len(self.image_paths)
    
    # Getter function 
    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert('RGB') 
        label = self.labels[index]

        if self.transform:
            image = self.transform(image)

        return image, label 


# Adjust transforamtion for the image 
# to be suitable for the ResNet18 model
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
]) 
    
# Creat Dataset
train_dataset = X_Ray_Dataset(root_dir= os.path.join(main_path, "train"), transform=transform)
test_dataset = X_Ray_Dataset(root_dir= os.path.join(main_path, "test"), transform=transform)
val_dataset = X_Ray_Dataset(root_dir= os.path.join(main_path, "val"), transform=transform)

# Load Dataset 
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Create trained ResNet18 model. 
# The training is done using IMAGENET1K_V1 which contain 1000 classes
# Already trained models facilitates the training and parameter 
# tuning instead of training from scrath.
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

# Change to 2 output classes 
model.fc = nn.Linear(model.fc.in_features, 2)  

# Pass the model to the computing resource (GPU or CPU)
model = model.to(device)

# Setting loss function and optimizer 
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001) # Optimize all neurons 

# Number of epoch
epoch_n = 10 

# Iterate over epoch number to train the model 
# using train dataset
for epoch in range(epoch_n):
    #====================== Perform Training =====================#
    model.train() # Switch to training mode 
    running_loss = 0.0 # Reset loss
    
    # Iterate over batch data
    for images, labels in train_loader:
        images = images.to(device) # Pass images to GPU/CPU
        labels = labels.to(device) # Pass labels to GPU/CPU

        optimizer.zero_grad() # Reset gradient
        output = model(images) # Predicted labels
        loss = criterion(output, labels) # Measure error between actual and predicted labels
        loss.backward() # Perform backpropagation 
        optimizer.step() # Update new parameters

        running_loss += loss.item() # Accumelated Loss - added .item() to prevent memory leak

    print(f"Epoch {epoch}/{epoch_n}, Loss : {running_loss/len(train_loader)}")
    
    #==================== Perform Validtion ====================# 
    model.eval() # Switch to evaluation mode 
    val_Labels = [] 
    val_preds = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images) # Get predicted labels
            _, preds = torch.max(outputs, 1) # Get label index [0,1]
            
            # Append preds and labels after passing it to
            # CPU and convert it to numpy array to be 
            # compatible to accuracy_score method 
            val_Labels.extend(labels.cpu().numpy())
            val_preds.extend(preds.cpu().numpy())

    val_accuracy = accuracy_score(val_Labels, val_preds)
    print('Validation accuracy:', val_accuracy)


#====================== Perform Testing =====================#
model.eval() # Switch to evaluation mode 
test_Labels = [] 
test_preds = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        
        outputs = model(images) # Get predicted labels
        _, preds = torch.max(outputs, 1) # Get label index [0,1]
        
        test_Labels.extend(labels.cpu().numpy())
        test_preds.extend(preds.cpu().numpy())

test_accuracy = accuracy_score(test_Labels, test_preds)
print('Test accuracy:', test_accuracy)

# Save - fixed typo in variable name
torch.save(model.state_dict(), 'pneumonia_classifier.pth')