import torch

def train(model, trainloader, criterion, optimizer, device, epochs=10):
    model.train()
    
    for epoch in range(epochs):
        running_loss = 0.0
        
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss:.3f}")