import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision.transforms import transforms
from torch.utils.data import DataLoader, ConcatDataset
from torch.optim.lr_scheduler import StepLR
import torchvision
import matplotlib.pyplot as plt
import argparse
import copy
from PIL import Image
from sklearn.metrics import confusion_matrix
import seaborn as sb
import pandas as pd
import time

# argument parser
parser = argparse.ArgumentParser()

# -save command line arg
parser.add_argument('-save', action='store_true', help='Saves the model after training')

# -load command line arg
parser.add_argument('-loadVal', action='store_true', help='Loads the saved model on validation data')

# -load command line arg
parser.add_argument('-loadTest', action='store_true', help='Loads the saved model on test data')

# -graph command line arg
parser.add_argument('-graph', action='store_true', help='Displays a training loss vs validation loss graph after training')

# -v command line arg
parser.add_argument('-v', action='store_true', help='Enables visualization of some example data')

# -img command line arg
parser.add_argument('-img', help="Predicts an image's class based on provided image folder path")

# -cm command line arg
parser.add_argument('-cm', action='store_true', help='Creates a confusion matrix for the loaded model (requires loadVal or loadTest)')

args = parser.parse_args()

img_path = args.img

# Define Multi-Layered Perceptron (MLP) model
class MLP(nn.Module):
    ''' Multi-Layer Perceptron model '''
    def __init__(self):
        super(MLP, self).__init__()
        self.flatten = nn.Flatten() # Flatten 2D image to 1D array
        self.fc1 = nn.Linear(64*64, 1024) # inputs to network. (creates the weights)
        self.bn1 = nn.BatchNorm1d(1024) # normalises values
        self.dropout = nn.Dropout(p=0.3)
        self.fc2 = nn.Linear(1024, 1024) # first hidden layer
        self.bn2 = nn.BatchNorm1d(1024) # normalises values
        self.fc3 = nn.Linear(1024, 1024)
        self.bn3 = nn.BatchNorm1d(1024) # normalises values
        self.fc4 = nn.Linear(1024, 15) # second hidden layer
          
    def forward(self, x):
        ''' Forward pass '''
        # Batch x of shape (B, C, W, H)
        x = self.flatten(x) # Shape: (B, 3072)
        x = F.relu(self.bn1(self.fc1(x))) # Shape: (B, 1024). activation function applied to inputs of 1st HL ---> produces outputs of 1st HL
        x = self.dropout(x)
        x = F.relu(self.bn2(self.fc2(x))) # Shape: (B, 1024). Applies ReLU to inputs of 2nd HL ---> produces outputs of 2nd HL
        x = self.dropout(x)
        x = F.relu(self.bn3(self.fc3(x)))
        x = self.dropout(x)
        x = self.fc4(x) # Shape: (B, 10). produces values at output layer
        return x

# Define training model
def train(net, trainloader, criterion, optimizer, device):
    net.train() # model set to training mode
    running_loss = 0.0 # used to calculate the loss across batches
    for data in trainloader:
        inputs, labels = data   # retrieves the inputs and labels from data
        inputs, labels = inputs.to(device), labels.to(device) # sends to device
        optimizer.zero_grad() # sets all gradients to 0
        outputs = net(inputs) # Get predictions (calls net.forward)
        loss = criterion(outputs, labels) # calculates loss (compares predicted outputs to true outputs)     
        loss.backward() # propagates the loss backwards through the network
        optimizer.step() # Update the weights in the network
        running_loss += loss.item() # Updates the loss
    avg_loss = running_loss/len(trainloader) # calculates average loss of training data
    return avg_loss


# Define testing model
def test(net, testloader, criterion, device):
    net.eval() # model set to evaluation mode
    correct = 0
    total = 0
    running_loss = 0.0
    with torch.no_grad(): # Do not calculate gradients
        for data in testloader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device) # Sends to device
            outputs = net(inputs) # Get predictions (calls net.forward)
            loss = criterion(outputs, labels) # calculates loss (compares predicted outputs to true outputs)    
            running_loss += loss.item() # Updates the loss
            _, pred = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (pred == labels).sum().item() # calculates how many are correct
    accuracy = correct/total # calculates accuracy of predictions
    avg_loss = running_loss/len(testloader) # calculates average loss of testing data
    return accuracy, avg_loss 


# Print out some example images for visualization
def visualize(train_loader):
    # Retrieves the examples from the train_loader
    examples = enumerate(train_loader)
    _, (example_data, example_targets) = next(examples)
    CATEGORIES = ["ac_src", "ammeter", "battery", "cap", 
                  "curr_src", "dc_volt_src_1", "dc_volt_src_2", 
                  "dep_curr_src", "dep_volt", "diode", "gnd_1", 
                  "gnd_2", "inductor", "resistor", "voltmeter"
                ]
    plt.figure()
    for i in range(6):
        plt.subplot(2,3,i+1)
        plt.tight_layout()
        plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
        plt.title("Ground Truth: {}".format(CATEGORIES[example_targets[i]]))
        plt.xticks([])
        plt.yticks([])
    plt.show()


# Loads the specified model
def load(mlp, test_loader, criterion, device):
    print("Loading params...")
    try:
        mlp.load_state_dict(torch.load('./models/mlp.pth'))
        print("Done!")
        if args.cm:
            test_acc, test_loss, cm = conf_matrix(mlp, test_loader, criterion, device)
            print(f"Test accuracy = {test_acc*100:.2f}%, Test loss = {test_loss:.4f}")
            CATEGORIES = ["ac_src", "ammeter", "battery", "cap", 
                  "curr_src", "dc_volt_src_1", "dc_volt_src_2", 
                  "dep_curr_src", "dep_volt", "diode", "gnd_1", 
                  "gnd_2", "inductor", "resistor", "voltmeter"
                ]
            df_cm = pd.DataFrame(cm, index = [i for i in CATEGORIES], columns = [i for i in CATEGORIES])
            sb.heatmap(df_cm, annot=True, fmt="")
            plt.xlabel("Predicted Labels")
            plt.ylabel("Ground Truth")
            plt.title("Confusion Matrix")
            plt.show()
        else:
            test_acc, test_loss = test(mlp, test_loader, criterion, device)
            print(f"Test accuracy = {test_acc*100:.2f}%, Test loss = {test_loss:.4f}")
        
    except:
        print("No model or accuracy file found to load")
    # finally:
        # exit()


# Saves the specified model
def save(mlp, test_loader, criterion, device, best_loss, best_params):
    # Loads in current best model and checks accuracy
    try:
        mlp.load_state_dict(torch.load('./models/mlp.pth'))
        _, saved_loss = test(mlp, test_loader, criterion, device)
    except:
        saved_loss = 1000
    # If accuracy of current model is better than saved model then overwrite saved model
    if (best_loss < saved_loss):
        torch.save(best_params,"./models/mlp.pth") # Saves best model
        print("Model saved to models/mlp.pth") 
    else:
        print("Saved model achieves lower validation loss. Discarding current model.")

# Predicts a single user input
def predictImage(mlp, device):
    print("Loading image prediction ....")
    CATEGORIES = ["ac_src", "ammeter", "battery", "cap", 
                "curr_src", "dc_volt_src_1", "dc_volt_src_2", 
                "dep_curr_src", "dep_volt", "diode", "gnd_1", 
                "gnd_2", "inductor", "resistor", "voltmeter"
                ]
    try:
        # Transform sequence for a single image
        transform = transforms.Compose([
            transforms.Resize((64,64)),
            transforms.ToTensor(), # 0-255 to 0-1
            transforms.Grayscale(), # converts rgb to grayscale
            transforms.Normalize([0.5], [0.5]) # Grayscale
        ])
        
        # Load the input image
        img = Image.open(img_path)
        
        # Convert to a bmp file
        index_of_last_dot = img_path.rfind(".")
        if index_of_last_dot != -1:  # Make sure the dot was found
            str_to_dot = img_path[:index_of_last_dot]
        file_out = str_to_dot + ".bmp" # Rename with .bmp extension
        img.save(file_out) # Save binary image
        img = Image.open(file_out) # Open bmp image to be dealt with
        
        img_tensor = transform(img).unsqueeze(0) # Add an extra dimension for batch size
        
        mlp.load_state_dict(torch.load('./models/mlp.pth')) # Load the saved model

        # Set model to evaluation mode
        mlp.eval()
        
        # Move image tensor to device
        img_tensor = img_tensor.to(device)
        
        # Get model predictions
        with torch.no_grad(): # Do not calculate gradients
            outputs = mlp(img_tensor)
                
        # Get predicted class index
        _, predicted = torch.max(outputs.data, 1)
        class_index = predicted.item()
        
        # Get predicted class label
        predicted_label = CATEGORIES[class_index]
        
        print("\nPrediction = ", predicted_label,"\n") # Indicate prediction
        
    except:
        print("Image not able to be loaded. Please ensure the provided image path is correct.")
    
    finally:
        exit()

# Confusion matrix during the test 
def conf_matrix(net, testloader, criterion, device):
    net.eval()
    correct = 0
    total = 0
    running_loss = 0.0
    y_true = []
    y_pred = []
    with torch.no_grad():
        for data in testloader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device) # Sends to device
            outputs = net(inputs) # Get predictions (calls net.forward)
            loss = criterion(outputs, labels) # calculates loss (compares predicted outputs to true outputs)  
            running_loss += loss.item() # Updates the loss  
            _, pred = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (pred == labels).sum().item() # calculates how many are correct
            y_true += labels.tolist()
            y_pred += pred.tolist()
    accuracy = correct/total # calculates accuracy of predictions
    avg_loss = running_loss/len(testloader) # calculates average loss of testing data
    c_matrix = confusion_matrix(y_true, y_pred)
    return accuracy, avg_loss, c_matrix
    

def main():
    
    # Transform sequence
    transform_normal = transforms.Compose([
        transforms.Resize((64,64)),
        transforms.ToTensor(), # 0-255 to 0-1
        transforms.Grayscale(), # converts rgb to grayscale
        transforms.Lambda(lambda x: 1-x), # symbols drawn in white. Converts to black with white background
        transforms.Normalize([0.5], [0.5]) # normalize
    ])
    
    # Transform sequence rotating by 90 degrees
    transform_90_degrees = transforms.Compose([
        transforms.Resize((64,64)),
        transforms.ToTensor(), # 0-255 to 0-1
        transforms.Grayscale(), # converts rgb to grayscale
        transforms.Lambda(lambda x: 1-x), # symbols drawn in white. Converts to black with white background
        transforms.RandomRotation(degrees=(90,90)),
        transforms.Normalize([0.5], [0.5]) # normalize
    ])
    
    # Transform sequence rotating by 180 degrees
    transform_180_degrees = transforms.Compose([
        transforms.Resize((64,64)),
        transforms.ToTensor(), # 0-255 to 0-1
        transforms.Grayscale(), # converts rgb to grayscale
        transforms.Lambda(lambda x: 1-x), # symbols drawn in white. Converts to black with white background
        transforms.RandomRotation(degrees=(180,180)),
        transforms.Normalize([0.5], [0.5]) # normalize
    ])
    
    # Transform sequence rotating by 270 degrees
    transform_270_degrees = transforms.Compose([
        transforms.Resize((64,64)),
        transforms.ToTensor(), # 0-255 to 0-1
        transforms.Grayscale(), # converts rgb to grayscale
        transforms.Lambda(lambda x: 1-x), # symbols drawn in white. Converts to black with white background
        transforms.RandomRotation(degrees=(270,270)),
        transforms.Normalize([0.5], [0.5]) # normalize
    ])
    
    # Transform sequence for test data
    transform_test = transforms.Compose([
        transforms.Resize((64,64)),
        transforms.ToTensor(),
        transforms.Grayscale(),
        transforms.Normalize([0.5], [0.5])
    ])
    
    BATCH_SIZE = 32
    train_path = "./data/train/"
    val_path = "./data/validation/"
    test_path = "./data/test"

    # Training data
    train_data_1 = torchvision.datasets.ImageFolder(train_path, transform=transform_normal)
    train_data_2 = torchvision.datasets.ImageFolder(train_path, transform=transform_90_degrees)
    train_data_3 = torchvision.datasets.ImageFolder(train_path, transform=transform_180_degrees)
    train_data_4 = torchvision.datasets.ImageFolder(train_path, transform=transform_270_degrees)
    train_data = ConcatDataset([train_data_1, train_data_2, train_data_3, train_data_4])

    # Validation data
    val_data_1 = torchvision.datasets.ImageFolder(val_path, transform=transform_normal)
    val_data_2 = torchvision.datasets.ImageFolder(val_path, transform=transform_90_degrees)
    val_data_3 = torchvision.datasets.ImageFolder(val_path, transform=transform_180_degrees)
    val_data_4 = torchvision.datasets.ImageFolder(val_path, transform=transform_270_degrees)
    val_data = ConcatDataset([val_data_1, val_data_2, val_data_3, val_data_4])
    
    # Test data
    test_data = torchvision.datasets.ImageFolder(test_path, transform=transform_test)

    # Dataloader for training
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)

    # Dataloader for validation
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)
    
    # Dataloader for testing
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)
    
    # Identify device to use
    device = ("cuda" if torch.cuda.is_available()
           else "mps" if torch.backends.mps.is_available()
           else "cpu")
    
    # Print out device used
    print(f"Using {device} device")
    
    # Initialise MLP and send parameters to the device
    mlp = MLP().to(device)

    # Enables user specified image to be predicted
    if args.img: predictImage(mlp, device)
    
    # Enables example images to be visualized
    if args.v: visualize(train_loader)
    
    # Define loss function
    loss_fn = nn.CrossEntropyLoss() # appropriate for classification models
    
    # Loads on the model on validation data
    if args.loadVal: 
        load(mlp, val_loader, loss_fn, device)
        return
    
    # Loads the model on test data
    if args.loadTest: 
        load(mlp, test_loader, loss_fn, device)
        return


    # -------------------------------- TRAIN THE MODEL --------------------------------------


    # optimizer parameters
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 1e-4

    # learning rate decay parameters
    STEP_SIZE=7
    GAMMA=0.1

    # Define optimizer
    optimizer = optim.Adam(mlp.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY) # Gets up to 65.21%

    # Define learning rate scheduler
    scheduler = StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)
            
    # tracks the best accuracy
    best_loss = 1000
    best_accuracy = 0
    best_params = None
    
    print("\nParameters:")
    print("-----------")
    print("-> Learning Rate: ", LEARNING_RATE)
    print("-> Learning Rate Decay: ", GAMMA)
    print("-> Learning Rate Decay Step: ", STEP_SIZE)
    print("-> Weight Decay: ", WEIGHT_DECAY)
    print("-> Batch Size: ", BATCH_SIZE)
    print("---------------------------------------")

    train_loss_array = []
    val_loss_array = []
    
    # Train the model for 20 epochs
    for epoch in range(20):
        train_loss = train(mlp, train_loader, loss_fn, optimizer, device)
        val_acc, val_loss = test(mlp, val_loader, loss_fn, device)
        
        # track losses over the epochs
        train_loss_array.append(train_loss)
        val_loss_array.append(val_loss)
        
        scheduler.step()
        print(f"Epoch {epoch+1}: Train loss = {train_loss:.4f}, Validation loss = {val_loss:.4f}, Validation accuracy = {val_acc:.4f}")
        
        # Finds the best accuracy and saves the model
        if (val_loss < best_loss):
            best_loss = val_loss
            best_accuracy = val_acc
            best_params = copy.deepcopy(mlp.state_dict()) # makes a deepcopy s.t. if mlp model changes, best_params won't
            
    print("---------------------------------------")
    print (f"Best validation accuracy = {best_accuracy*100:.2f}%  |   Best validation loss = {best_loss:.4f}")
        
    # Saves the model
    if args.save: save(mlp, val_loader, loss_fn, device, best_loss, best_params)
    
    # Plots the training loss vs validation loss
    if args.graph:
        plt.plot(train_loss_array, label='Training loss')
        plt.plot(val_loss_array, label='Validation loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss vs Validation Loss')
        plt.legend()
        plt.show()


# Call the main method if executed properly
if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time = {execution_time:.3f} seconds")
    # plt.show()