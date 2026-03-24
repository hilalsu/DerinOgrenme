# Odev 1 - CIFAR-10 KNN

Bu odevde CIFAR-10 veri seti diskten okunarak K-Nearest Neighbor (KNN)
siniflandiricisi from-scratch uygulanmistir.

## Calistirma

```bash
python main.py
```

## Ornek Cikti

```text
Train shape: (5000, 3072), Labels: (5000,)
Test  shape: (1000, 3072), Labels: (1000,)

KNN Sonuclari (CIFAR-10)
-----------------------------------
k = 1  | accuracy = 0.2680 (26.80%)
k = 3  | accuracy = 0.2610 (26.10%)
k = 5  | accuracy = 0.2660 (26.60%)
k = 7  | accuracy = 0.2740 (27.40%)
-----------------------------------
En iyi sonuc: k = 7 -> accuracy = 0.2740
```
