import torch
from src.data.dataset import load_dataset
from src.models.train import train
from src.models.test import test

classes = ('CN', 'AD', 'MCI')

id2label = {i: classes[i] for i in range(len(classes))}
label2id = {classes[i]: i for i in range(len(classes))}

trainset, testset, valset = load_dataset(label2id=label2id)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=0)
testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=0)

train(trainloader)
test(testloader)
