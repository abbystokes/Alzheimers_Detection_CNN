import torch
import torch.nn as nn
import torch.optim as optim
from src.data.dataset import load_dataset
from src.models.models import SNeurodCNN

def train(trainloader):

    net = SNeurodCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.0001)

    max_epochs = 10

    for epoch in range(max_epochs):

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs.float())
            loss = criterion(outputs, torch.Tensor(labels))
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 10 == 9:    # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 10:.3f}')
                running_loss = 0.0

        if epoch % 10 == 9:
            # Saving the final epoch of the model
            PATH = f'./models/sneurod_cnn_{epoch+1}.pth'
            torch.save(net.state_dict(), PATH)

    print('Finished Training')
