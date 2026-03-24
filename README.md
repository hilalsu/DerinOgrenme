# Derin Öğrenme Ders Ödevleri

Bu repo, ders ödevlerini tek bir yerde düzenli şekilde toplamak amacıyla oluşturulmuştur.

### Klasör Yapısı

* `data/` : Ortak veri setleri (CIFAR-10 burada tutulur)
* `odev1/` : CIFAR-10 üzerinde KNN sınıflandırma ödevi
* `odev2/` : Fashion-MNIST üzerinde Neural Network ödevi

### CIFAR-10 Kurulumu

1. CIFAR-10 Python sürümünü indir:
   [https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz](https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz)
2. Arşivi açın.
3. `cifar-10-batches-py` klasörünü `data/` altına koyun:

   ```
   data/cifar-10-batches-py
   ```

### Ödev 1 Çalıştırma

```bash
cd odev1
pip install numpy
python main.py
```

**Not:** KNN from-scratch yazıldığı için tam veri ile yavaş çalışabilir. Varsayılan alt-küme kullanılır:

* `train_limit = 5000`
* `test_limit = 1000`

### Ödev 2 Çalıştırma

```bash
cd odev2
pip install torch numpy
python main.py
```

**Not:** Veri seti `data/fashion-mnist` altına otomatik indirilir ve diskten okunur.

---

