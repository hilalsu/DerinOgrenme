# Odev 2 - Fashion-MNIST Neural Network

Bu odevde Fashion-MNIST veri seti diskten okunur, normalize edilir ve
basit bir yapay sinir agi (1 hidden layer) ile siniflandirma yapilir.

## Dosyalar

- `data_loader.py`: Veri indirme/diskten okuma ve DataLoader olusturma
- `model.py`: Basit Neural Network modeli
- `main.py`: Egitim + test akisi

Kullanilan cihaz: cpu
Indiriliyor: train-images-idx3-ubyte.gz
Indiriliyor: train-labels-idx1-ubyte.gz
Indiriliyor: t10k-images-idx3-ubyte.gz
Indiriliyor: t10k-labels-idx1-ubyte.gz
Epoch 1/5 | Train Loss: 0.5975 | Train Acc: 79.64%
Epoch 2/5 | Train Loss: 0.4182 | Train Acc: 85.35%
Epoch 3/5 | Train Loss: 0.3833 | Train Acc: 86.36%
Epoch 4/5 | Train Loss: 0.3562 | Train Acc: 87.18%
Epoch 5/5 | Train Loss: 0.3354 | Train Acc: 87.98%

Test Sonuclari
Test Loss: 0.3635
Final Test Accuracy: 87.02%

## Calistirma

```bash
cd odev2
pip install torch numpy
python main.py
```

Not: Veri seti `data/fashion-mnist` altina otomatik indirilir ve diskten okunur.
