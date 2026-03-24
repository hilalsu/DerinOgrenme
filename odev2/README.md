# Odev 2 - Fashion-MNIST Neural Network

Bu odevde Fashion-MNIST veri seti diskten okunur, normalize edilir ve
basit bir yapay sinir agi (1 hidden layer) ile siniflandirma yapilir.

## Dosyalar

- `data_loader.py`: Veri indirme/diskten okuma ve DataLoader olusturma
- `model.py`: Basit Neural Network modeli
- `main.py`: Egitim + test akisi

## Calistirma

```bash
cd odev2
pip install torch numpy
python main.py
```

Not: Veri seti `data/fashion-mnist` altina otomatik indirilir ve diskten okunur.
