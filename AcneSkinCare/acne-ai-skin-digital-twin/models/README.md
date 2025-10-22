# HAM10000 Model Directory

Bu klasöre eğitilmiş HAM10000 modelini koyun.

## Beklenen Dosya

- `ham10000_efficientnetb3.keras` - EfficientNetB3 modeli (Stage B - fine-tuned)

## Model Özellikleri

- **Input Size:** 300x300x3
- **Classes:** 7 sınıf
- **Architecture:** EfficientNetB3 + Dense layers
- **Training:** Focal Loss + MixUp + Class Weighting
- **Dataset:** HAM10000 (10,015 dermatoskopi görüntüsü)

## Sınıflar

1. `akiec` - Actinic Keratoses (Güneş Lekesi) - 327 örnek
2. `bcc` - Basal Cell Carcinoma (Bazal Hücreli Kanser) - 514 örnek
3. `bkl` - Benign Keratosis-like Lesions (İyi Huylu) - 1,099 örnek
4. `df` - Dermatofibroma (Deri Fibroma) - 115 örnek
5. `mel` - Melanoma (Kötü Huylu Kanser) - 1,113 örnek
6. `nv` - Nevus (Ben/Mole) - 6,705 örnek (%67)
7. `vasc` - Vascular Lesions (Damar Lezyonları) - 142 örnek

## Kullanım

```python
from src.ai.predict import analyze_image

result = analyze_image('path/to/image.jpg', 'models/ham10000_efficientnetb3.keras')
print(result['predicted_class'])      # 'mel', 'nv', etc.
print(result['confidence'])           # 0.85
print(result['predicted_class_desc']) # 'Melanoma (Kötü Huylu Kanser)'
```

## Colab'dan Model İndirme

Eğitim tamamlandığında Colab'dan indirin:

```python
# Colab'da
from google.colab import files
files.download('b3_stageB.keras')
```

Sonra bu klasöre `ham10000_efficientnetb3.keras` olarak kaydedin.