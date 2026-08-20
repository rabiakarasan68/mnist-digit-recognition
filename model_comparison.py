import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neural_network import MLPClassifier

def veriyi_yukle():
    digits = load_digits()
    X = digits.data
    y = digits.target

    print(f"Toplam örnek sayısı: {X.shape[0]}")
    print(f"Her görüntünün boyutu: 8x8 = {X.shape[1]} piksel")
    print(f"Sınıflar (rakamlar) : {np.unique(y)}")

    fig, axes = plt.subplots(2, 5, figsize=(10,4))
    for i, ax in enumerate(axes.flat):
        ax.imshow(digits.images[i], cmap="gray")
        ax.set_title(f"Etiket: {digits.target[i]}")
        ax.axis("off")
    plt.tight_layout()
    plt.savefig("ornek_rakamlar.png", dpi=120)
    print("\n Örnek rakamlar 'ornek_rakamlar.png' olarak kaydedildi.\n")

    return X, y

def veriyi_hazirla(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"Eğitim seti : {X_train.shape[0]} örnek")
    print(f"Test seti : {X_test.shape[0]} örnek\n")

    return X_train_scaled, X_test_scaled, y_train, y_test

def modelleri_egit(X_train, X_test, y_train, y_test):
    modeller = {
        "Lojistik Regresyon": LogisticRegression(max_iter=2000),
        "Destek Vektör Makinesi (SVM)": SVC(kernel="rbf", gamma="scale"),
        "Rastgele Orman": RandomForestClassifier(n_estimators=200, random_state=42),
        "Basit Sinir Ağı (MLP)": MLPClassifier(
            hidden_layer_sizes=(64, 32), max_iter=500, random_state=42
        ),
    }

    sonuclar = {}
    en_iyi_model = None
    en_iyi_dogruluk = 0
    en_iyi_isim = ""

    print("=" * 50)
    print("Model Karşılaştırması")
    print("=" * 50)

    for isim, model in modeller.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        dogruluk = accuracy_score(y_test, y_pred)
        sonuclar[isim] = dogruluk
        print(f"{isim:35s}: %{dogruluk*100:.2f} doğruluk")

        if dogruluk > en_iyi_dogruluk:
            en_iyi_dogruluk = dogruluk
            en_iyi_model = model
            en_iyi_isim = isim

    print("=" * 50)
    print(f"🏆 En iyi model: {en_iyi_isim} (%{en_iyi_dogruluk*100:.2f})")
    print("=" * 50 + "\n")

    return en_iyi_model, en_iyi_isim, X_test, y_test
    
def detayli_degerlendirme(model, isim, X_test, y_test):
    y_pred = model.predict(X_test)

    print(f"'{isim}' modeli için sınıflandırma raporu:\n")
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(7,6))
    plt.imshow(cm, cmap="Blues")
    plt.title(f"Karışıklık Matrisi - {isim}")
    plt.xlabel("Tahmin Edilen Rakam")
    plt.ylabel("Gerçek Rakam")
    plt.xticks(range(10))
    plt.yticks(range(10))
    for i in range(10):
        for j in range(10):
            plt.text(j, i, cm[i,j], ha="center", va="center",
                    color="white" if cm[i,j] > cm.max() / 2 else "black")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig("karisiklik_matrisi.png", dpi=120)
    print("\n✓ Karışıklık matrisi 'karisiklik_matrisi.png' olarak kaydedildi.")

if __name__ == "__main__":
    X, y = veriyi_yukle()
    X_train, X_test, y_train, y_test = veriyi_hazirla(X, y)
    en_iyi_model, en_iyi_isim, X_test, y_test = modelleri_egit(
        X_train, X_test, y_train, y_test
    )
    detayli_degerlendirme(en_iyi_model, en_iyi_isim, X_test, y_test)
