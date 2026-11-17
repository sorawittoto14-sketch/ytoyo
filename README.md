# ytoyo — ตรวจจับเสียงอินฟราซาวด์ของช้างด้วย SincNet-LSTM

โมเดลตรวจจับเสียง rumble (อินฟราซาวด์) ของช้างจากเสียงดิบ 1 มิติ ตามขั้นตอนในโปสเตอร์ ELSENSE:

```
Raw Audio → Sliding Window (3 s, overlap 1 s) → Balanced Dataset → SincNet (10–60 Hz) → LSTM → ผลตรวจจับ
```

> ⚠️ **โมเดลใน `models/sinc_lstm.pt` เทรนด้วยเสียงช้างจำลอง (synthetic)** เพราะข้อมูลเสียงช้างจริงที่มี label
> (เช่น Cornell Elephant Listening Project) ต้องขอสิทธิ์ก่อนใช้ ตัวเลขด้านล่างจึงยังไม่ใช่ผลกับเสียงจริงในภาคสนาม —
> ให้เทรนต่อ (fine-tune) ด้วยเสียงจริงก่อนนำไปใช้งาน (ดูหัวข้อ “เทรนด้วยข้อมูลจริง”)

## โมเดล

| ขั้น | รายละเอียด |
|---|---|
| Input | เสียงดิบ resample เป็น 250 Hz, หน้าต่าง 3 วินาที (750 samples) |
| SincNet | ฟิลเตอร์ band-pass แบบ sinc 32 ตัว เรียนรู้ความถี่ตัดล่าง/บนเอง แต่ถูกบังคับให้อยู่ใน **10–60 Hz** |
| Conv | Conv1D 2 ชั้น ลดเหลือ 5 เฟรม/วินาที |
| LSTM | BiLSTM 2 ชั้น (hidden 64) ให้ความน่าจะเป็นทุก 0.2 s → บอกได้ทั้ง “มีเสียงช้างไหม” และ “เริ่ม–จบเมื่อไร” |
| Output | คะแนนหน้าต่าง = max ของคะแนนเฟรม |

Loss = BCE ระดับหน้าต่าง + BCE ระดับเฟรม (เมื่อรู้ตำแหน่งเสียงช้าง เช่นจากข้อมูลจำลองหรือตาราง Raven) · พารามิเตอร์ ~197k

## ผลลัพธ์ (ชุดทดสอบเสียงจำลอง 4,000 หน้าต่าง, สมดุล 50/50)

| | โมเดลนี้ (เสียงจำลอง) | โปสเตอร์ (เสียงจริง) |
|---|---|---|
| Accuracy | 89.6 % | 83.12 % |
| Precision | 93.5 % | 85.40 % |
| Recall | 85.1 % | 81.25 % |
| F1 / ROC-AUC | 0.891 / 0.962 | – |

ข้อมูลต่างกัน จึง **เทียบกันตรง ๆ ไม่ได้** เพราะเสียงจำลองอาจง่ายกว่าเสียงจริง

- Recall แยกตาม SNR (10–60 Hz): −6…−2 dB 66 % · −2…2 dB 86 % · 2…6 dB 91 % · 6…12 dB 93 %
- False positive ตามเสียงรบกวน: ไม่มีสิ่งรบกวน 3.6 % · ลม 3.7 % · แมลง 6.7 % · ฟ้าร้อง 9.1 % · เครื่องยนต์ 11.4 % · เครื่องบิน 12.8 %
- **ไฟล์เสียงยาว 30 นาที (มีเสียงช้าง 40 ครั้ง)**: ตรวจพบ 38/40 (95 %), เวลาเริ่ม/จบคลาดเคลื่อนเฉลี่ย (median) ~0.2 s, แต่ยังมี false alarm ~82 ครั้ง/ชม. — ต้องปรับ threshold กับเสียงจริงในพื้นที่

ดูรายละเอียดทั้งหมดใน `models/sinc_lstm.metrics.json` (รวมช่วงความถี่ที่ SincNet เรียนรู้ได้)

## การใช้งาน

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt

# ตรวจจับเสียงช้างในไฟล์เสียง (wav/flac, sample rate ใดก็ได้, ยาวเป็นวันได้)
python -m elephant_rumble.detect recording.wav --out-dir detections/
#   → detections/recording.rumbles.txt  (Raven selection table เปิดดูใน Raven Pro ได้)
#   ปรับ --threshold (ค่าเริ่มต้น 0.6) และ --min-dur (1.0 s) ให้เหมาะกับพื้นที่

# เทรนใหม่ด้วยเสียงจำลอง (~7 นาทีบน CPU 4 คอร์)
python -m elephant_rumble.train --n-train 20000 --n-val 2000 --n-test 4000 --epochs 30

# ทดสอบตัวตรวจจับกับไฟล์ยาวจำลอง
python -m elephant_rumble.demo_long --minutes 30 --rumbles 40

python -m pytest -q tests
```

## เทรนด้วยข้อมูลจริง

1. มีไฟล์เสียง + ตาราง Raven (`Begin Time (s)`, `End Time (s)`) ที่ระบุเสียงช้าง ให้ตัดเป็นคลิปแบบสมดุล:
   ```python
   from elephant_rumble.data import clips_from_raven
   clips_from_raven("site1.wav", "site1.selections.txt", "data/real_clips")   # ทำซ้ำได้หลายไฟล์
   ```
   จะได้ `data/real_clips/pos/`, `neg/` และ `segments.json` (ตำแหน่งเสียงช้างในแต่ละคลิป → label รายเฟรม)
   หรือจะเตรียมคลิป 3 วินาทีใส่โฟลเดอร์ `pos/` และ `neg/` เองก็ได้
2. Fine-tune จากโมเดลเสียงจำลอง:
   ```bash
   python -m elephant_rumble.train --data data/real_clips --init models/sinc_lstm.pt --lr 5e-4 --out models/sinc_lstm_real.pt
   ```

## โครงสร้างไฟล์

```
elephant_rumble/
  config.py     ค่าคงที่ (250 Hz, หน้าต่าง 3 s / hop 2 s, ย่าน 10–60 Hz, threshold)
  synth.py      สร้างเสียงช้างจำลอง (1–6 s, f0 10–30 Hz + ฮาร์มอนิก) และเสียงรบกวน (ลม ฟ้าร้อง เครื่องยนต์ เครื่องบิน แมลง)
  data.py       Dataset: เสียงจำลอง, โฟลเดอร์ pos/neg, ตาราง Raven
  model.py      SincConv + SincLSTM
  train.py      เทรน/ประเมินผล
  detect.py     สแกนไฟล์ยาวแบบ sliding window → ตาราง Raven
  demo_long.py  ประเมินระดับเหตุการณ์บนไฟล์ยาวจำลอง
models/sinc_lstm.pt (+ .metrics.json)
```
