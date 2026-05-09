import torch
import matplotlib.pyplot as plt

from RNN.dataset_loader import OCRDataset
from RNN.decode import decode
from RNN.model import CRNN

def decode_label(label, chars):
    return "".join(chars[i-1] for i in label if i != 0)

model = CRNN(num_classes=37)
model.load_state_dict(torch.load("crnn.pth"))
model.eval()

dataset = OCRDataset("data")
img, label = dataset[0]

with torch.no_grad():
    pred = model(img.unsqueeze(0))
    pred = pred.permute(1,0,2)

plt.imshow(img.squeeze(0), cmap='gray')

gt_text = decode_label(label, dataset.chars)
pred_text = decode(pred, dataset.chars)

plt.title(f"GT: {gt_text} | Pred: {pred_text}")
plt.axis('off')
plt.show()