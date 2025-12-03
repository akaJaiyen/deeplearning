# 🦋 โมเดลจำแนกสายพันธุ์ผีเสื้อ (Butterfly Species Classification)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]([YOUR_NOTEBOOK_URL_HERE](https://colab.research.google.com/drive/11BL48hsZnMJmIY1ETgTHlauv-aSQlJRO?usp=sharing))


## 📖 คำอธิบายโปรเจกต์ (Project Description)

โปรเจกต์นี้คือการสร้างโมเดล Deep Learning (Convolutional Neural Network) เพื่อจำแนกสายพันธุ์ผีเสื้อ 75 สายพันธุ์ โดยใช้เทคนิค **Transfer Learning** ด้วยโมเดล **MobileNetV2** (ที่ผ่านการฝึกสอนบนชุดข้อมูล ImageNet มาแล้ว)

โค้ดทั้งหมดถูกออกแบบมาให้ทำงานบน **Google Colab** โดยดึงชุดข้อมูลมาจาก **Kaggle API** โดยตรง ทำให้สามารถรันโปรเจกต์ซ้ำได้ง่าย

---

## 🗂 สารบัญ (Table of Contents)

* [เกี่ยวกับชุดข้อมูล (Dataset)](#dataset)
* [วิธีการรันโปรเจกต์ (How to Run)](#how-to-run)
* [ขั้นตอนการทำงานของโมเดล (Methodology)](#methodology)
* [ผลลัพธ์การฝึกโมเดล (Results)](#results)
* [เทคโนโลยีที่ใช้ (Tech Stack)](#tech-stack)

---

## <a name="dataset"></a>📦 เกี่ยวกับชุดข้อมูล (Dataset)

เราใช้ชุดข้อมูล [Butterfly Image Classification](https://www.kaggle.com/datasets/phucthaiv02/butterfly-image-classification) จาก Kaggle

### **ข้อควรทราบที่สำคัญมาก:**
ชุดข้อมูลนี้มีโครงสร้างแบบ **"Flat Structure"** ซึ่งหมายความว่า:
1.  ไฟล์รูปภาพทั้งหมด (เช่น `Image_1.jpg`, `Image_2.jpg`) ถูกเก็บไว้รวมกันในโฟลเดอร์ `train/` และ `test/`
2.  **ไม่มี** การแบ่งโฟลเดอร์ย่อยตามสายพันธุ์ผีเสื้อ
3.  "ฉลาก" (Label) หรือชื่อสายพันธุ์ของแต่ละภาพ ถูกเก็บแยกไว้ในไฟล์ `Training_set.csv`

ด้วยเหตุนี้ เราจึงไม่สามารถใช้ `image_dataset_from_directory` ของ Keras ได้โดยตรง แต่เราใช้ `pandas` เพื่ออ่านไฟล์ `.csv` แล้วจึงสร้าง `tf.data.Dataset` เพื่อโหลดรูปภาพตาม Path ที่ระบุใน CSV แทน (ดูรายละเอียดใน **Section 2** ของโน้ตบุ๊ก)

---

## <a name="how-to-run"></a>🚀 วิธีการรันโปรเจกต์ (How to Run)

คุณสามารถรันโปรเจกต์นี้ซ้ำได้ทั้งหมดบน Google Colab โดยทำตามขั้นตอนดังนี้:

### 1. สิ่งที่ต้องมี
* บัญชี Google (สำหรับ Google Colab)
* บัญชี Kaggle

### 2. การตั้งค่า Kaggle API (ทำเพียงครั้งเดียว)
โมเดลนี้ต้องใช้ Kaggle API เพื่อดาวน์โหลดชุดข้อมูลโดยตรง

1.  ไปที่ **Kaggle.com** > (รูปโปรไฟล์ของคุณ) > **Settings**
2.  เลื่อนลงมาที่ส่วน **API** แล้วคลิก `Create New API Token`
3.  ไฟล์ `kaggle.json` จะถูกดาวน์โหลดมา ให้เปิดไฟล์นั้นด้วย Text Editor
4.  คุณจะเห็น `{"username":"YOUR_USERNAME","key":"YOUR_API_KEY"}`

### 3. การตั้งค่า Google Colab Secrets
เพื่อความปลอดภัย เราจะไม่เก็บ API Key ไว้ในโค้ด แต่จะใช้ฟีเจอร์ "Secrets" ของ Colab

1.  ในหน้า Colab ของคุณ คลิกที่ **ไอคอนรูปกุญแจ 🔑** ในแถบเครื่องมือด้านซ้าย
2.  คลิก `+ Add a new secret`
3.  สร้าง Secret ตัวที่ 1:
    * **Name:** `KAGGLE_USERNAME`
    * **Value:** (ใส่ค่า `YOUR_USERNAME` ที่คัดลอกมา)
4.  สร้าง Secret ตัวที่ 2:
    * **Name:** `KAGGLE_KEY`
    * **Value:** (ใส่ค่า `YOUR_API_KEY` ที่คัดลอกมา)
5.  *สำคัญ:* เปิดสวิตช์ "Notebook access" (ปุ่มสีฟ้า) ให้กับ Secret ทั้งสองตัว

### 4. การตั้งค่า Runtime
1.  ไปที่เมนู `Runtime` > `Change runtime type`
2.  ในส่วน "Hardware accelerator" ให้เลือก **GPU** (สำคัญมากสำหรับการฝึกโมเดล)

### 5. รันโน้ตบุ๊ก (Run the Notebook)
รันเซลล์ (Cell) ทั้งหมดตามลำดับตั้งแต่บนลงล่าง:

* **Section 1: Setup and Data Import**
    * ติดตั้ง `kaggle` และตั้งค่า API จาก Colab Secrets
    * ดาวน์โหลดและแตกไฟล์ชุดข้อมูล (`.zip`)
* **Section 2: Data Exploration and Preprocessing**
    * **(สำคัญ)** อ่านไฟล์ `Training_set.csv` ด้วย `pandas`
    * แบ่งข้อมูลออกเป็น **Train (80%), Validation (10%), และ Test (10%)**
    * สร้าง `tf.data.Dataset` pipeline เพื่อโหลดและเตรียมรูปภาพ (ปรับขนาด, ทำ Augmentation)
* **Section 3: Building the Model**
    * สร้างโมเดลโดยใช้ **MobileNetV2** เป็น Base Model
    * เพิ่ม Classifier Head (GlobalAveragePooling, Dropout, Dense) ที่ส่วนท้าย
* **Section 4: Model Training**
    * **Phase 1 (Feature Extraction):** "แช่แข็ง" Base Model และฝึกเฉพาะส่วนหัว (10 Epochs)
    * **Phase 2 (Fine-Tuning):** "ละลาย" Base Model และฝึกต่อด้วย Learning Rate ต่ำๆ (อีก 10 Epochs)
* **Section 5: Evaluating the Results**
    * พล็อต_กราฟ Model Accuracy และ Model Loss เพื่อวิเคราะห์ประสิทธิภาพการฝึก
* **Section 6: Detailed Analysis**
    * ทดสอบโมเดลกับ **Test Set** (10% ที่เราแบ่งไว้)
    * แสดงตัวอย่างการทำนาย (Prediction)
    * แสดง **Confusion Matrix** และ **Classification Report**
* **Section 7: Conclusion**
    * สรุปผลลัพธ์ของโปรเจกต์

---

## <a name="methodology"></a>🧠 ขั้นตอนการทำงานของโมเดล (Methodology)

1.  **Data Pipeline:** เราใช้ `pandas.read_csv` เพื่อโหลดข้อมูล Path และ Label จากนั้นใช้ `tf.data.Dataset.from_tensor_slices` เพื่อสร้าง Pipeline ที่มีประสิทธิภาพสูงในการป้อนข้อมูลให้ GPU
2.  **Data Splitting:** เราใช้ `sklearn.model_selection.train_test_split` สองครั้งเพื่อแบ่งชุดข้อมูล `Training_set.csv` ออกเป็น 3 ส่วน (Train/Validation/Test) โดยใช้ `stratify=...` เพื่อให้แน่ใจว่าทุกสายพันธุ์ถูกกระจายไปในทุกชุดอย่างสมดุล
3.  **Data Augmentation:** เพื่อป้องกัน Overfitting เราใช้เทคนิค Augmentation กับชุดข้อมูล Training เท่านั้น (เช่น `RandomFlip`, `RandomRotation`, `RandomZoom`)
4.  **Transfer Learning:** เราใช้ MobileNetV2 ที่ผ่านการฝึกบน ImageNet มาแล้ว ซึ่งช่วยให้โมเดลเรียนรู้ Feature พื้นฐาน (เช่น ขอบ, ลวดลาย) ได้อย่างรวดเร็ว และเราเพียงแค่ต้อง "ปรับจูน" (Fine-Tuning) ให้มันรู้จัก Feature เฉพาะของผีเสื้อ

---

## <a name="results"></a>📊 ผลลัพธ์การฝึกโมเดล (Results)

โมเดลสุดท้ายทำงานได้ดีมาก โดยได้ผลลัพธ์บนชุดข้อมูล Validation (Validation Set) ดังนี้:

* **Validation Accuracy: ~89-90%**

<img width="1156" height="547" alt="image" src="https://github.com/user-attachments/assets/0fc36b0b-2d1b-4c76-b9b0-b86a207c0017" />


จากกราฟ:
* **กราฟ Loss (ด้านขวา):** `Training Loss` (สีฟ้า) และ `Validation Loss` (สีส้ม) ลดลงไปในทิศทางเดียวกัน ซึ่งเป็นสัญญาณที่ดีว่าโมเดลกำลัง "เรียนรู้เพื่อทำความเข้าใจ" (Generalizing) ไม่ใช่ "ท่องจำ" (Overfitting)
* **กราฟ Accuracy (ด้านซ้าย):** เส้น Training และ Validation อยู่ใกล้กันมาก ยืนยันว่าโมเดลมีประสิทธิภาพที่ดีและไม่ Overfit

---

## <a name="tech-stack"></a>💻 เทคโนโลยีที่ใช้ (Tech Stack)

* **Platform:** Google Colab (GPU)
* **Core Libraries:** Python 3, TensorFlow, Keras
* **Data Handling:** Pandas, Scikit-learn
* **Visualization:** Matplotlib, Seaborn
* **Utilities:** Kaggle API, NumPy



