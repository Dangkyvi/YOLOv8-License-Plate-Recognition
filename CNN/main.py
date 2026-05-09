import torch
import torch.nn as nn
import torch.optim as optim

from model import CNN
from data import load_data
from train import train
from test import test

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    trainloader, testloader = load_data()
    
    model = CNN().to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    train(model, trainloader, criterion, optimizer, device, epochs=10)
    test(model, testloader, device)

if __name__ == "__main__":
    main()