import torch
from tqdm import tqdm
from torchvision import transforms
from utils.dataloader import generate_TrainValTest_dataloaders
from utils.config import train_path, val_path, num_classes, epochs
from model.deep_emotion import DeepEmotion
import torch.optim as optim




if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# You can print the device to verify it's set to GPU
print("Device:", device)


def Train(epochs, trainloader, valloader, optimizer, model):
    print("===================================Start Training===================================")

    for epoch in range(epochs):
        train_loss = 0
        validation_loss = 0
        train_correct = 0
        val_correct = 0

        model.train()
        for data, labels in tqdm(trainloader, desc=f"Epoch {epoch + 1}/{epochs} (Training)"):
            data, labels = data.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(data)
            loss = model.compute_loss(outputs, labels)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            train_correct += torch.sum(preds == labels.data)


        model.eval()
        for data, labels in tqdm(valloader, desc=f"Epoch {epoch + 1}/{epochs} (Validation)"):
            data, labels = data.to(device), labels.to(device)
            val_outputs = model(data)

            val_loss = model.compute_loss(val_outputs, labels)

            validation_loss += val_loss.item()
            _, val_preds = torch.max(val_outputs, 1)
            val_correct += torch.sum(val_preds == labels.data)

        train_loss = train_loss / len(train_loader)
        train_acc = train_correct.double() / len(train_loader.dataset)
        validation_loss = validation_loss / len(val_loader)
        val_acc = val_correct.double() / len(val_loader.dataset)

        print('Epoch: {} \tTraining Loss: {:.8f} \tValidation Loss {:.8f} \tTraining Accuracy {:.3f}% \tValidation Accuracy {:.3f}%'
              .format(epoch + 1, train_loss, validation_loss, train_acc * 100, val_acc * 100))

    torch.save(model.state_dict(), 'checkpoints/deep_emotion-{}.pt'.format(epochs))
    print("===================================Training Finished===================================")


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=0.5, std=0.5)
])

train_loader, val_loader, _ = generate_TrainValTest_dataloaders(transform, train_path, val_path)

net = DeepEmotion(num_classes)
net = net.to(device)

optimizer = optim.SGD(net.parameters(), lr=0.005, momentum=0.9)

Train(epochs, train_loader, val_loader, optimizer, net)