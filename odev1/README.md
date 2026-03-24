# Odev 1 Raporu - CIFAR-10 KNN

## 1) Amaç

Bu çalışmada CIFAR-10 veri seti üzerinde from-scratch K-Nearest Neighbor (KNN) sınıflandırıcısı uygulanmıştır. Farklı k değerleri (1, 3, 5, 7) test edilmiş ve performansları karşılaştırılmıştır.

## 2) Yöntem

* Veri seti diskten pickle dosyaları (`data_batch_1..5`, `test_batch`) ile okunmuştur.
* Model **sklearn** kullanılmadan, **numpy** ile sıfırdan yazılmıştır.
* Uzaklık ölçümü olarak **Öklid mesafesi** kullanılmıştır.
* Tahmin adımında en yakın **k** komşunun çoğunluk oylaması alınmıştır.

## 3) Deney Ayarları

* Train alt-küme: 5000
* Test alt-küme: 1000
* Test edilen k değerleri: 1, 3, 5, 7

## 4) Sonuçlar

| k | Accuracy        |
| - | --------------- |
| 1 | 0.2680 (26.80%) |
| 3 | 0.2610 (26.10%) |
| 5 | 0.2660 (26.60%) |
| 7 | 0.2740 (27.40%) |

**En iyi sonuç:** k = 7 → accuracy = 0.2740

## 5) Analiz

* Küçük k değeri genelde daha esnek ancak gürültüye daha duyarlıdır.
* Büyük k değeri daha düzgün karar sınırı çizer fakat aşırı genelleme yapabilir.
* CIFAR-10 gibi yüksek boyutlu görüntü verilerinde temel KNN’in doğruluğu sınırlıdır; özellik çıkarımı veya daha gelişmiş modellerle performans artırılabilir.
* Deney sonuçlarına göre en iyi doğruluk **k=7** ile elde edilmiştir (%27.40).

## 6) Sonuç

Bu ödevde KNN algoritması temel prensipleriyle uygulanmış, farklı k değerlerinin doğruluk üzerindeki etkisi gözlemlenmiştir. Temel KNN’in CIFAR-10 gibi karmaşık veri setlerinde sınırlı doğruluk sağladığı görülmüştür; ileri seviye özellik çıkarımı veya derin öğrenme tabanlı modellerle performans artırılabilir.

---
