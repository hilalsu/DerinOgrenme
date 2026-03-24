# derinogrenme

Bu repo, ders odevlerini tek bir yerde duzenli toplamak icin olusturuldu.

## Klasor yapisi

- `data/`: Ortak veri setleri (CIFAR-10 burada tutulur)
- `odev1/`: CIFAR-10 uzerinde KNN siniflandirma odevi
- `odev2/`: Sonraki odeve ayrilmis klasor

## CIFAR-10 kurulumu

1. CIFAR-10 Python surumunu indir:
   - https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
2. Arsivi ac.
3. `cifar-10-batches-py` klasorunu `data/` altina koy:
   - `data/cifar-10-batches-py`

## Odev1 calistirma

```bash
cd odev1
pip install numpy
python main.py
```

Not: KNN from-scratch oldugu icin tam veri ile yavas calisabilir. Varsayilan olarak
alt-kume (`train_limit=5000`, `test_limit=1000`) kullanilir.
