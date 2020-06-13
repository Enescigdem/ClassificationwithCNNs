import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import itertools
def plot_confusion_matrix(cm, classes,
                          cmap=plt.cm.Blues):
   

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    threshold = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j]),horizontalalignment="center",color="white" if cm[i, j] > threshold else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')


data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
}

dataset_directory = 'Houses-test'
image_datasets = {x: datasets.ImageFolder(os.path.join(dataset_directory, x),data_transforms[x])for x in ['train', 'val', 'test']}
#Batch size is set as 64 
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=64,
                                             shuffle=True, num_workers=0)
              for x in ['train', 'val' ,'test']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val','test']}

print({x: len(image_datasets[x]) for x in ['train', 'val','test']})
class_names = image_datasets['train'].classes

device = "cuda:0" if torch.cuda.is_available() else "cpu"

def train_model(model, criterion, optimizer, scheduler, epoch_number):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_train_acc = 0.0
    best_val_acc = 0.0
    best_test_acc = 0.0
    train_acc_history = list()
    val_acc_history = list()
    test_acc_history = list()
    for epoch in range(epoch_number):
        print('Epoch {}/{}'.format(epoch, epoch_number - 1))
        
        # Each epoch has a training and validation phase
        for part in ['train', 'val', 'test']:
            if part == 'train':
                scheduler.step()
                model.train()  
            else:
                model.eval()  

            current_loss = 0.0
            current_phase_correct_outputnumber = 0
            # For each phase in datasets are iterated
            for inputs, labels in dataloaders[part]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward
                with torch.set_grad_enabled(part == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    # Backpropagate and opitimize Training part
                    if part == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                current_loss += loss.item() * inputs.size(0)
                current_phase_correct_outputnumber += torch.sum(preds == labels.data)

            current_loss = current_loss / dataset_sizes[part]
            epoch_acc = current_phase_correct_outputnumber.double() / dataset_sizes[part]

            if part == 'val':
                val_acc_history.append(epoch_acc)
            elif part == 'test':
                test_acc_history.append(epoch_acc)
            else:
                train_acc_history.append(epoch_acc)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                part, current_loss, epoch_acc))

            # deep copy the model
            if part == 'train' and epoch_acc > best_train_acc:
                best_train_acc = epoch_acc
            if part == 'val' and epoch_acc > best_val_acc:
                best_val_acc = epoch_acc
            if part == 'test' and epoch_acc > best_test_acc:
                best_test_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_difference = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_difference // 60, time_difference % 60))
    #Printed best accuracies
    print('Best train Acc: {:4f}'.format(best_train_acc))
    print('Best validation Acc: {:4f}'.format(best_val_acc))
    print('Best test Acc: {:4f}'.format(best_test_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    #Plot accuracy graph 
    plt.xlabel("Training Epochs")
    plt.ylabel("Accuracy")
    plt.plot(train_acc_history, color="green")
    plt.plot(val_acc_history, color="yellow")
    plt.plot(test_acc_history, color="red")
    plt.gca().legend(('Train', 'Validation', 'Test'))

    plt.show()

    return model

#resnet18
#For scratch learning, pretrained=false is done
training_model = models.resnet18(pretrained=True)
num_ftrs = training_model.fc.in_features
training_model.fc = nn.Linear(num_ftrs, 15) 

for param in training_model.fc.parameters():
    param.requires_grad = False

'''
for param in training_model.layer1.parameters():
    param.requires_grad = False
for param in training_model.layer2.parameters():
    param.requires_grad = False
for param in training_model.layer3.parameters():
    param.requires_grad = False
for param in training_model.layer4.parameters():
    param.requires_grad = False
'''

#Alexnet
'''
training_model = models.alexnet(pretrained=True)

#Uncomment for feature extraction
for param in training_model.parameters():
    param.requires_grad = False

num_ftrs = training_model .classifier[6].in_features
training_model .classifier[6] = nn.Linear(num_ftrs, 15)
'''


training_model  = training_model .to(device)
criterion = nn.CrossEntropyLoss()

optimizer_ft = optim.SGD(training_model.parameters(), lr=0.1, momentum=0.9)

# Decayed learning rate value by gamma value once for every 7 step
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
######################################################################


#main call

trained_model = train_model(training_model, criterion, optimizer_ft, exp_lr_scheduler,epoch_number=1)

#Output class number is 15
#Getting confusion matrix values
confusion_matrixx = torch.zeros(15, 15)
np.set_printoptions(precision=2)
# Plot non-normalized confusion matrix
with torch.no_grad():
    for i, (inputs, classes) in enumerate(dataloaders['test']):
        inputs = inputs.to(device)
        classes = classes.to(device)
        outputs = trained_model(inputs)
        _, preds = torch.max(outputs, 1)
        
        for t, p in zip(classes.view(-1), preds.view(-1)):
            confusion_matrixx[t.long(), p.long()] += 1
#Plot size is set
plt.figure(figsize = (15,15))
plot_confusion_matrix(confusion_matrixx,classes=class_names)
plt.show()



