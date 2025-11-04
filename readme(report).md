# Final Project Report

## การจำแนกสายพันธุ์ผีเสื้อด้วย Deep Learning (Butterfly Species Classification)

**รายวิชา:** การเรียนรู้เชิงลึก (Deep Learning) 204466  
**สมาชิก:**  
- พิจักษณ์ ลิ้มล้ำเลิศกุล (6610502161)  
- กิรณา รักประกอบกิจ (6610505284)

---

## 1. หัวข้อและความน่าสนใจของโครงงาน

ผีเสื้อมีความหลากหลายของลวดลายและสีสันสูง การจำแนกด้วยตาเปล่ามักต้องอาศัยผู้เชี่ยวชาญ Deep Learning สามารถเพิ่มความแม่นยำและลดเวลาได้  
โครงงานนี้ใช้ **Convolutional Neural Network (CNN)** เพื่อให้ระบบเรียนรู้ลักษณะเฉพาะของผีเสื้อ เช่น ลวดลาย สี และโครงสร้างร่างกาย

---

## 2. เหตุผลที่เลือกใช้ Deep Learning

การจำแนกภาพผีเสื้อเป็นปัญหาที่ซับซ้อน วิธีการดั้งเดิมอย่าง HOG, SIFT, LBP และโมเดล SVM หรือ Random Forest มีข้อจำกัด เช่น ต้องออกแบบ Feature เองและไม่สามารถจับความสัมพันธ์เชิงพื้นที่ได้ดี

### 2.1 ข้อดีของ Deep Learning

CNN สามารถเรียนรู้ Feature ได้เองจากข้อมูลภาพโดยไม่ต้องออกแบบล่วงหน้า อีกทั้งยังใช้ **Transfer Learning** จากโมเดลที่ผ่านการฝึกบน ImageNet ทำให้เรียนรู้ได้เร็วและใช้ข้อมูลน้อยลง

---

## 3. สถาปัตยกรรมของโมเดล (Model Architecture)

### 3.1 ภาพรวมของโครงสร้าง
ใช้ **MobileNetV2** เป็น Base Model (Feature Extractor)  
และสร้าง **Classifier Head** ใหม่สำหรับจำแนกผีเสื้อ 75 ชนิด

### 3.2 รายละเอียดของ MobileNetV2
- ใช้ *Depthwise Separable Convolution* เพื่อลดจำนวนพารามิเตอร์  
- มี *Inverted Residual Block* ซึ่งประกอบด้วย Expansion → Depthwise → Projection  
- ใช้ **Batch Normalization** และ **ReLU6** เพื่อความเสถียร

### 3.3 ส่วนหัวจำแนก (Classifier Head)
```
GlobalAveragePooling2D → Dropout(0.3) → Dense(128, ReLU) → Dense(75, Softmax)
```

### 3.4 ขั้นตอนการฝึก (Training Phases)
1. **Phase 1:** แช่แข็ง MobileNetV2 ฝึกเฉพาะ Classifier Head  
2. **Phase 2:** Fine-tuning บางชั้นของ MobileNetV2 ด้วย learning rate ที่เล็กลง

### 3.5 Activation & Hyperparameters
| Parameter | Value |
|------------|--------|
| Activation | ReLU6 / Softmax |
| Optimizer | Adam |
| Loss | Categorical Crossentropy |
| Batch Size | 32 |
| Epochs | 20 (10 + 10 Fine-tuning) |

---

## 4. การอธิบายโค้ด (Code Explanation)

- **Section 1:** Setup & Data Import (ติดตั้ง Kaggle API, โหลดข้อมูล)  
- **Section 2:** Data Preprocessing (resize, normalize, augment)  
- **Section 3:** Model Building (โหลด MobileNetV2, เพิ่ม Head)  
- **Section 4:** Model Training (Adam, Crossentropy, Accuracy)  
- **Section 5:** Evaluation & Visualization (กราฟ accuracy/loss)  
- **Section 6:** Test & Confusion Matrix  
- **Section 7:** Conclusion

---

## 5. ข้อมูลชุดข้อมูล (Dataset Description)

- แหล่งที่มา: [Kaggle – Butterfly Image Classification](https://www.kaggle.com/datasets/phucthaiv02/butterfly-image-classification)  
- จำนวนภาพ: ~38,000 ภาพ  
- จำนวนสายพันธุ์: 75 ชนิด  
- ขนาดภาพ: 224×224 px  
- การเตรียมข้อมูล: Normalize, One-hot encoding, Augmentation

---

## 6. การเทรนโมเดล (Model Training)

| ขั้นตอน | รายละเอียด |
|----------|--------------|
| แบ่งข้อมูล | Train 80%, Validation 10%, Test 10% |
| Preprocessing | Resize 224×224, Normalize [0,1] |
| Augmentation | Random Flip, Rotation, Zoom |
| Batch / Shuffle | Batch 32, Shuffle 1000 |
| Optimizer | Adam |
| Learning Rate | 0.001 (Phase 1), 0.00001 (Phase 2) |
| Epochs | 20 (10 + 10 Fine-tuning) |

---

## 7. การประเมินผล (Evaluation)

- **Training Accuracy:** ~90%  
- **Validation Accuracy:** ~89–90%  
- **Test Accuracy:** ~89%  
- **F1-Score (เฉลี่ย):** 0.88  
- **Training/Validation Loss:** ลดลงต่อเนื่อง (~1.2 → ~0.25)  
- ไม่มีสัญญาณ overfitting — กราฟ train/validation ใกล้กัน

> โมเดลสามารถ generalize ได้ดีและมีความเสถียรในการเรียนรู้

---

## 8. บทความอ้างอิง

1. Sandler, M. et al. *MobileNetV2: Inverted Residuals and Linear Bottlenecks.* CVPR 2018  
2. Chollet, F. *Deep Learning with Python.* Manning, 2017  
3. [Kaggle Dataset: Butterfly Image Classification](https://www.kaggle.com/datasets/phucthaiv02/butterfly-image-classification)  
4. [TensorFlow Transfer Learning Guide](https://www.tensorflow.org/tutorials/images/transfer_learning)

---

## 9. สรุปผล (Conclusion)

| Metric | ผลลัพธ์ | หมายเหตุ |
|--------|----------|-----------|
| Training Accuracy | ≈ 90% | โมเดลเรียนรู้ดี ไม่ overfit |
| Validation Accuracy | ≈ 89–90% | แม่นยำในข้อมูลใหม่ |
| Test Accuracy | ≈ 89% | ประสิทธิภาพใกล้เคียง validation |
| F1-Score | ≈ 0.88 | ผลสม่ำเสมอในทุกคลาส |
| Epochs | 20 | ใช้เวลาเทรน ~40 นาที (GPU Colab) |

โมเดลสามารถจำแนกสายพันธุ์ผีเสื้อได้แม่นยำสูงและมีการเรียนรู้ที่เสถียร การใช้ Transfer Learning + Fine-tuning ช่วยเพิ่มความแม่นยำและลดเวลาเทรนได้อย่างชัดเจน

---

## 10. การแบ่งงาน

| สมาชิก | หน้าที่ | สัดส่วน |
|---------|----------|----------|
| พิจักษณ์ ลิ้มล้ำเลิศกุล | ออกแบบและฝึกโมเดล, วิเคราะห์ผล | 50% |
| กิรณา รักประกอบกิจ | เขียนรายงาน, สร้าง README.md, สรุปผล | 50% |
