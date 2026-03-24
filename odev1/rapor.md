# Odev 1 Raporu - CIFAR-10 KNN

## 1) Amac

Bu calismada CIFAR-10 veri seti uzerinde from-scratch K-Nearest Neighbor (KNN)
siniflandiricisi uygulanmistir. Farkli `k` degerleri (`1, 3, 5, 7`) test edilmis
ve performanslari karsilastirilmistir.

## 2) Yontem

- Veri seti diskten pickle dosyalari (`data_batch_1..5`, `test_batch`) ile okunmustur.
- Model sklearn kullanmadan, numpy ile sifirdan yazilmistir.
- Uzaklik olcumu olarak Oklid mesafesi kullanilmistir.
- Tahmin adiminda en yakin `k` komsunun cogunluk oylamasi alinmistir.

## 3) Deney Ayarlari

- Train alt-kume: `5000`
- Test alt-kume: `1000`
- Test edilen k degerleri: `1, 3, 5, 7`

## 4) Sonuclar

Asagidaki tabloya kendi calistirma ciktinizi yazin:

| k | Accuracy |
|---|---|
| 1 | ... |
| 3 | ... |
| 5 | ... |
| 7 | ... |

## 5) Analiz

- Kucuk `k` degeri genelde daha esnek ama gurultuye daha duyarli olur.
- Buyuk `k` degeri daha duzgun karar siniri cizer fakat asiri genelleme yapabilir.
- En iyi sonucu veren `k` degeri, veri dagilimina ve secilen alt-kume buyuklugune baglidir.

## 6) Sonuc

Bu odevde KNN algoritmasi temel prensipleriyle uygulanmis, farkli `k` degerlerinin
dogruluk uzerindeki etkisi gozlemlenmistir.
