import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

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
    
def veri_yukleyicileri_hazirla(batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    egitim_seti = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_seti = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

    egitim_yukleyici = DataLoader(egitim_seti, batch_size=batch_size, shuffle=True)
    test_yukleyici = DataLoader(test_seti, batch_size=1000, shuffle=False)

    return egitim_yukleyici, test_yukleyici

def egit(model, cihaz, egitim_yukleyici, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(egitim_yukleyici):
        data, target = data.to(cihaz), target.to(cihaz)

        optimizer.zero_grad()
        cikis = model(data)
        kayip = F.cross_entropy(cikis, target)
        kayip.backward()
        optimizer.step()

        if batch_idx % 200 == 0:
            print(f" Epoch {epoch} [{batch_idx * len(data)}/{len(egitim_yukleyici.dataset)}]"
                  f"Kayıp(Loss): {kayip.item():.4f}")
            
def test_et(model, cihaz, test_yukleyici):
    model.eval()
    test_kaybi = 0
    dogru = 0
    with torch.no_grad():
        for data, target in test_yukleyici:
            data, target = data.to(cihaz), target.to(cihaz)
            cikis = model(data)
            test_kaybi += F.cross_entropy(cikis, target, reduction="sum").item()
            tahmin = cikis.argmax(dim=1)
            dogru += tahmin.eq(target).sum().item()

    test_kaybi /= len(test_yukleyici.dataset)
    dogruluk = 100.00 * dogru / len(test_yukleyici.dataset)
    print(f"\nTest sonucu -> Ortalama kayıp: {test_kaybi:.4f}, "
          f"Doğruluk: {dogru}/{len(test_yukleyici.dataset)} (%{dogruluk:.2f})\n")
    return dogruluk

def main():
    cihaz = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Kullanılan cihaz: {cihaz}\n")

    egitim_yukleyici, test_yukleyici = veri_yukleyicileri_hazirla()

    model = RakamCNN().to(cihaz)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    EPOCH_SAYISI = 5
    en_iyi_dogruluk = 0

    for epoch in range(1, EPOCH_SAYISI + 1):
        print(f"--- Epoch {epoch}/{EPOCH_SAYISI} ---")
        egit(model, cihaz, egitim_yukleyici, optimizer, epoch)
        dogruluk = test_et(model, cihaz, test_yukleyici)

        if dogruluk > en_iyi_dogruluk:
            en_iyi_dogruluk = dogruluk
            torch.save(model.state_dict(), "en_iyi_model.pth")
            print("Yeni en iyi model kaydedildi (en_iyi_model.pth)\n")

    print(f"Eğitim tamamlandı. En iyi doğruluk: %{en_iyi_dogruluk:.2f}")

    ornekleri_gorsellestir(model, cihaz, test_yukleyici)

def ornekleri_gorsellestir(model, cihaz, test_yukleyici):
    model.eval()
    data, target = next(iter(test_yukleyici))
    data, target = data.to(cihaz), target.to(cihaz)

    with torch.no_grad():
        tahminler = model(data).argmax(dim=1)
    
    fig, axes = plt.subplots(2, 5, figsize=(10, 4))
    for i, ax in enumerate(axes.flat):
        goruntu = data[i].cpu().squeeze()
        ax.imshow(goruntu, cmap="gray")
        renk = "green" if tahminler[i] == target[i] else "red"
        ax.set_title(f"Tahmin: {tahminler[i].item()}", color=renk)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig("cnn_tahmin_ornekleri.png", dpi=120)
    print("Tahmin örnekleri 'cnn_tahmin_ornekleri.png' olarak kaydedildi.")

if __name__ == "__main__":
    main()
