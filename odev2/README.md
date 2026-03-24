# Fashion-MNIST Neural Network Ödevi

Bu ödevde Fashion-MNIST veri seti diskten okunur, normalize edilir ve basit bir yapay sinir ağı (1 hidden layer) ile sınıflandırılır.

### Dosyalar

* `data_loader.py` : Veri indirme/diskten okuma ve DataLoader oluşturma
* `model.py` : Basit Neural Network modeli
* `main.py` : Eğitim ve test akışı

### Çalıştırma

```bash
cd odev2
pip install torch numpy
python main.py
```

**Not:** Veri seti `data/fashion-mnist` altına otomatik indirilir ve diskten okunur.

### Örnek Eğitim Çıktısı

```
Epoch 1/5 | Train Loss: 0.5975 | Train Acc: 79.64%
Epoch 2/5 | Train Loss: 0.4182 | Train Acc: 85.35%
Epoch 3/5 | Train Loss: 0.3833 | Train Acc: 86.36%
Epoch 4/5 | Train Loss: 0.3562 | Train Acc: 87.18%
Epoch 5/5 | Train Loss: 0.3354 | Train Acc: 87.98%
```

### Test Sonuçları

```
Test Loss: 0.3635
Final Test Accuracy: 87.02%
```

---

Bunu yapayım mı?
