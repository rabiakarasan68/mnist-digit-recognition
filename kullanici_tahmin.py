import tkinter as tk
from PIL import Image, ImageDraw
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

# CNN MODELİ
class RakamCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)

        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)

        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        x = self.dropout1(x)

        x = torch.flatten(x, 1)

        x = F.relu(self.fc1(x))

        x = self.dropout2(x)

        x = self.fc2(x)

        return x

# MODELİ YÜKLE
cihaz = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = RakamCNN().to(cihaz)

model.load_state_dict(
    torch.load(
        "en_iyi_model.pth",
        map_location=cihaz
    )
)

model.eval()

# GÖRÜNTÜ DÖNÜŞTÜRME
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# ÇİZİM ALANI
PENCERE_BOYUTU = 280

pencere = tk.Tk()
pencere.title("MNIST Rakam Tahmini")
pencere.resizable(False, False)

canvas = tk.Canvas(
    pencere,
    width=PENCERE_BOYUTU,
    height=PENCERE_BOYUTU,
    bg="black"
)

canvas.pack(padx=10, pady=10)

goruntu = Image.new(
    "L",
    (PENCERE_BOYUTU, PENCERE_BOYUTU),
    0
)

cizim = ImageDraw.Draw(goruntu)

def ciz(event):

    x = event.x
    y = event.y

    kalinlik = 20

    canvas.create_oval(
        x - kalinlik // 2,
        y - kalinlik // 2,
        x + kalinlik // 2,
        y + kalinlik // 2,
        fill="white",
        outline="white"
    )

    cizim.ellipse(
        [
            x - kalinlik // 2,
            y - kalinlik // 2,
            x + kalinlik // 2,
            y + kalinlik // 2
        ],
        fill=255
    )


canvas.bind("<B1-Motion>", ciz)

def temizle():

    canvas.delete("all")

    cizim.rectangle(
        [0, 0, PENCERE_BOYUTU, PENCERE_BOYUTU],
        fill=0
    )

    sonuc_label.config(text="Tahmin: -")

def tahmin_et():

    # 280x280 görüntüyü 28x28'e küçült
    img = goruntu.resize((28, 28))

    # PIL -> Tensor
    tensor = transform(img)

    # Batch boyutu ekle
    tensor = tensor.unsqueeze(0)

    tensor = tensor.to(cihaz)

    # Model tahmini
    with torch.no_grad():

        cikis = model(tensor)

        olasiliklar = F.softmax(cikis, dim=1)

        tahmin = torch.argmax(cikis, dim=1).item()

        guven = olasiliklar[0][tahmin].item() * 100

    sonuc_label.config(
        text=f"Tahmin: {tahmin}   Güven: %{guven:.2f}"
    )

buton_frame = tk.Frame(pencere)
buton_frame.pack(pady=5)

tahmin_butonu = tk.Button(
    buton_frame,
    text="Tahmin Et",
    command=tahmin_et,
    width=15
)

tahmin_butonu.grid(
    row=0,
    column=0,
    padx=5
)


temizle_butonu = tk.Button(
    buton_frame,
    text="Temizle",
    command=temizle,
    width=15
)

temizle_butonu.grid(
    row=0,
    column=1,
    padx=5
)


sonuc_label = tk.Label(
    pencere,
    text="Tahmin: -",
    font=("Arial", 18)
)

sonuc_label.pack(pady=10)

pencere.mainloop()
