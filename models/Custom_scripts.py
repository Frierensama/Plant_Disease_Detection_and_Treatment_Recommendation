import torch
import torch.nn as nn
import pickle 

class CustomNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.feature_extract = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3 , padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(16, 32, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2, stride=2),


            nn.Conv2d(32, 64, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Dropout(0.2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),

            nn.Linear(64*28*28, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128,64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.4),

            nn.Linear(64,num_classes)
        )

    def forward(self,X):
        out = self.feature_extract(X)
        out = self.classifier(out)

        return out
    

def train_block(model, train_dataloader, criterion, optimizer, device):
    model.to(device)
    model.train()

    total_batch_loss = 0.0
    correctly_predicted = 0
    total = 0

    for images, labels in train_dataloader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

        total_batch_loss += loss.item()
        total += len(labels)
        _,predicted = torch.max(output, 1)
        correctly_predicted += (predicted == labels).sum().item()

    epoch_loss = total_batch_loss/len(train_dataloader)
    acc = correctly_predicted/total

    return epoch_loss, acc


def val_block(model, valid_dataloader, criterion, device):
    model.to(device)
    model.eval()

    total_batch_loss = 0.0
    correctly_predicted = 0
    total = 0

    with torch.no_grad():
        for images, labels in valid_dataloader:
            images = images.to(device)
            labels = labels.to(device)


            output = model(images)
            loss = criterion(output, labels)

            total_batch_loss += loss.item()
            total += len(labels)
            _,predicted = torch.max(output, 1)
            correctly_predicted += (predicted == labels).sum().item()

    epoch_loss = total_batch_loss/len(valid_dataloader)
    acc = correctly_predicted/total

    return epoch_loss, acc