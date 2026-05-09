from torch.utils.data import DataLoader
import torch
from RNN.dataset_loader import OCRDataset
import torch.nn as nn

from RNN.model import CRNN

ctc_loss = nn.CTCLoss(blank=0)

dataset = OCRDataset("data")
loader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=lambda x:x)

model = CRNN(num_classes=37)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(5):
    for batch in loader:
        imgs = torch.stack([b[0] for b in batch])
        labels = [b[1] for b in batch]

        preds = model(imgs)
        preds = preds.permute(1,0,2)

        input_lengths = torch.full((len(batch),),preds.size(0), dtype=torch.long)
        target_lengths = torch.tensor([len(l) for l in labels])

        targets = torch.cat(labels)
        loss = ctc_loss(preds.log_softmax(2), targets,input_lengths, target_lengths)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch}, Loss {loss.item()}")

torch.save(model.state_dict(), "crnn.pth")