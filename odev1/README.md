# CIFAR-10 KNN Ödevi

Bu ödevde CIFAR-10 veri seti diskten okunarak K-Nearest Neighbor (KNN) sınıflandırıcısı baştan uygulanmıştır.

### Çalıştırma

```bash
python main.py
```

### Örnek Çıktı

```
Train shape: (5000, 3072), Labels: (5000,)
Test  shape: (1000, 3072), Labels: (1000,)

KNN Sonuçları (CIFAR-10)
-----------------------------------
k = 1  | accuracy = 0.2680 (26.80%)
k = 3  | accuracy = 0.2610 (26.10%)
k = 5  | accuracy = 0.2660 (26.60%)
k = 7  | accuracy = 0.2740 (27.40%)
-----------------------------------
En iyi sonuç: k = 7 -> accuracy = 0.2740
```

### Sonuçların Yorumlanması

* En iyi doğruluk `k=7` ile elde edilmiştir (%27.40).
* `k=1` daha esnek olup tekil ve gürültülü örneklere duyarlıdır.
* `k` arttıkça karar sınırı daha düzgün hale gelir.
* CIFAR-10 gibi yüksek boyutlu görüntü verilerinde temel KNN’in doğruluğu sınırlıdır; özellik çıkarımı veya daha gelişmiş modellerle performans artırılabilir.

---

