import torch
from src.models.models import SNeurodCNN
from sklearn.metrics import classification_report


def test(testloader):
    PATH = './models/sneurod_cnn_10.pth'
    net = SNeurodCNN()
    net.load_state_dict(torch.load(PATH))

    correct = 0
    total = 0

    y_true = []
    y_pred = []

    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            # calculate outputs by running images through the network
            outputs = net(images.float())
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)

            y_true.extend(labels)
            y_pred.extend(predicted)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # print(f'Accuracy of the network on the 1000 test images: {100 * correct // total} %')

    print(classification_report(y_true, y_pred))
