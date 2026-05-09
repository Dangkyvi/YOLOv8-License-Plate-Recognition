from PIL import Image, ImageDraw
import random, os

chars = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
def random_plate():
    return "".join(random.choices(chars, k=7))
os.makedirs("data", exist_ok=True)

for i in range(2000):
    text = random_plate()
    img = Image.new("L",(128,32),color=255)
    draw = ImageDraw.Draw(img)
    draw.text((5,5), text, fill=0)

    img.save(f"data/{text}_{i}.png")
    