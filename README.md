# ytoyo — ตรวจจับเสียงอินฟราซาวด์ของช้างด้วย SincNet-LSTM

โมเดลตรวจจับเสียง rumble (อินฟราซาวด์) ของช้างจากเสียงดิบ 1 มิติ ตามขั้นตอนในโปสเตอร์ ELSENSE:

```
Raw Audio → Sliding Window (3 s, overlap 1 s) → Balanced Dataset → SincNet (10–60 Hz) → LSTM → ผลตรวจจับ
```

มี 2 โมเดล:

| ไฟล์ | เทรนด้วย | ใช้เมื่อ |
|---|---|---|
| `models/sincnet_lstm_real_elephant_data.pt` | **เสียงช้างจริง** (Cornell ELP, ช้างป่าแอฟริกา ประเทศคองโก) เริ่มจากโมเดลเสียงจำลอง | **ค่าเริ่มต้น ใช้ตัวนี้** |
| `models/sinc_lstm.pt` | เสียงช้างจำลอง (synthetic) | ใช้เป็นจุดเริ่มต้นสำหรับ fine-tune |

## โมเดล

| ขั้น | รายละเอียด |
|---|---|
| Input | เสียงดิบ resample เป็น 250 Hz, หน้าต่าง 3 วินาที (750 samples) |
| SincNet | ฟิลเตอร์ band-pass แบบ sinc 32 ตัว เรียนรู้ความถี่ตัดล่าง/บนเอง แต่ถูกบังคับให้อยู่ใน **10–60 Hz** |
| Conv | Conv1D 2 ชั้น ลดเหลือ 5 เฟรม/วินาที |
| LSTM | BiLSTM 2 ชั้น (hidden 64) ให้ความน่าจะเป็นทุก 0.2 s → บอกได้ทั้ง “มีเสียงช้างไหม” และ “เริ่ม–จบเมื่อไร” |
| Output | คะแนนหน้าต่าง = max ของคะแนนเฟรม |

Loss = BCE ระดับหน้าต่าง + BCE ระดับเฟรม (เมื่อรู้ตำแหน่งเสียงช้าง เช่นจากข้อมูลจำลองหรือตาราง Raven) · พารามิเตอร์ ~197k

## ข้อมูลจริง (Cornell Elephant Listening Project)

- **เสียง**: ไฟล์บันทึก 24 ชม. จากเครือข่ายเครื่องบันทึกเสียงอุทยาน Nouabalé-Ndoki ประเทศคองโก (8 kHz) เปิดให้ใช้ฟรีบน AWS Open Data
  `s3://congo8khz-pnnn` — ต้องให้เครดิต “Cornell Elephant Listening Project”
- **Label**: ตาราง Raven ที่ผู้เชี่ยวชาญระบุตำแหน่ง rumble จาก repo `statlogic/Elephant-Rumble-Detection` → **3,938 rumbles ใน 163 ไฟล์ (163 วัน)**
- ดาวน์โหลดเฉพาะช่วงเวลาที่ต้องใช้ด้วย HTTP Range (~91 MB แทนที่จะเป็น ~225 GB): `python -m elephant_rumble.elp --out data/elp` (~5 นาที)
- ตัวอย่างลบ (ไม่มีช้าง) สุ่มจากไฟล์เดียวกัน ห่างจาก rumble ที่ label ไว้ ≥ 10 วินาที จำนวนเท่ากับตัวอย่างบวก (Balanced)
- **แบ่ง 80/10/10 ตามไฟล์ (วันบันทึก)** เพื่อไม่ให้เสียงวันเดียวกันอยู่ทั้งใน train และ test:

| ชุด | ไฟล์ (วัน) | rumbles | หน้าต่างไม่มีช้าง |
|---|---|---|---|
| Train | 131 | 3,276 | 3,276 |
| Val | 16 | 297 | 297 |
| Test | 16 | 365 | 365 |

## ผลลัพธ์บนเสียงจริง (ชุด Test — 16 วันที่โมเดลไม่เคยเห็น)

**ระดับหน้าต่าง 3 วินาที (730 หน้าต่าง, สมดุล)**

| โมเดล | Accuracy | Precision | Recall | AUC |
|---|---|---|---|---|
| `sincnet_lstm_real_elephant_data.pt` (จำลอง → fine-tune เสียงจริง) | **96.3 %** | **97.5 %** | **95.1 %** | 0.991 |
| เทรนเสียงจริงอย่างเดียว (ไม่ใช้เสียงจำลอง) | 96.9 % | 98.9 % | 94.8 % | 0.992 |
| `sinc_lstm.pt` (เสียงจำลองอย่างเดียว) | 71.0 % | 89.2 % | 47.7 % | 0.806 |
| โปสเตอร์ ELSENSE (ข้อมูลของโปสเตอร์) | 83.12 % | 85.40 % | 81.25 % | – |

→ เสียงจำลองอย่างเดียวไม่พอ (recall 48 %) ต้องเทรนด้วยเสียงจริง · ข้อมูลต่างชุดกับโปสเตอร์ จึงเทียบกันตรง ๆ ไม่ได้

**ระดับการใช้งานจริง: สแกนไฟล์ 24 ชม. เต็มวัน 3 วันจากชุด Test (72 ชม., 203 rumbles)**

| threshold | ตรวจพบ rumble | false alarm / ชม. |
|---|---|---|
| 0.5 | 96.1 % | 10.2 |
| 0.6 | 95.1 % | 6.5 |
| 0.7 | 94.1 % | 4.3 |
| **0.8 (ค่าเริ่มต้น)** | **93.1 %** | **2.5** |
| 0.9 | 92.1 % | 1.2 |

เวลาเริ่ม/จบคลาดเคลื่อน (median) ~0.2–0.6 วินาที · บาง “false alarm” อาจเป็นเสียงช้างจริงที่ไม่ได้ label ไว้ · ประเมินซ้ำได้ด้วย
`elephant_rumble.elp.evaluate_day(model, filename)` · ผลละเอียด: `models/sincnet_lstm_real_elephant_data.metrics.json`, `models/sincnet_lstm_real_elephant_data.day_eval.json`

> ⚠️ ข้อมูลเป็น **ช้างป่าแอฟริกา (forest elephant)** ในป่าคองโก ช้างเอเชียในไทยและเสียงรบกวนในพื้นที่ (รถ เครื่องสูบน้ำ ฯลฯ) ต่างกัน
> ควร fine-tune ต่อด้วยเสียงที่อัดจากอุปกรณ์ ELSENSE ในพื้นที่จริง และปรับ threshold ก่อนใช้แจ้งเตือน

## การใช้งาน

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt

# ตรวจจับเสียงช้างในไฟล์เสียง (wav/flac, sample rate ใดก็ได้, ยาวเป็นวันได้)
python -m elephant_rumble.detect recording.wav --out-dir detections/
#   → detections/recording.rumbles.txt  (Raven selection table เปิดดูใน Raven Pro ได้)
#   ใช้ models/sincnet_lstm_real_elephant_data.pt, ปรับ --threshold (ค่าเริ่มต้น 0.8) และ --min-dur (1.0 s) ให้เหมาะกับพื้นที่

# ดาวน์โหลดเสียงช้างจริง + เทรน (80/10/10)
python -m elephant_rumble.elp --out data/elp
python -m elephant_rumble.train --data elp:data/elp --init models/sinc_lstm.pt --lr 1e-3 --out models/sincnet_lstm_real_elephant_data.pt

# เทรนด้วยเสียงจำลอง (20,000 / 2,500 / 2,500)
python -m elephant_rumble.train

# ทดสอบตัวตรวจจับกับไฟล์ยาวจำลอง
python -m elephant_rumble.demo_long --minutes 30 --rumbles 40

python -m pytest -q tests
```

## ONNX (สำหรับอุปกรณ์ / Edge)

`models/sincnet_lstm_real_elephant_data.onnx` — แปลงจากโมเดลเสียงช้างจริง (ผลตรงกับ PyTorch, ต่างกัน < 1e-6)

| | ชื่อ | รูปร่าง |
|---|---|---|
| input | `audio` | float32 `(batch, 750)` = เสียง 3 วินาที @ 250 Hz |
| output | `frame_probs` | float32 `(batch, 15)` = ความน่าจะเป็นมีเสียงช้าง ทุก 0.2 วินาที |
| output | `window_prob` | float32 `(batch,)` = มีเสียงช้างในหน้าต่างนี้ไหม |

```python
import onnxruntime as ort
sess = ort.InferenceSession("sincnet_lstm_real_elephant_data.onnx")
frame_probs, window_prob = sess.run(None, {"audio": windows})   # windows: (N, 750) float32
```
ก่อนส่งเข้าโมเดล: ใช้ช่องแรก → low-pass + resample เป็น 250 Hz → ตัดหน้าต่าง 3 วินาที ทุก 2 วินาที
ข้อมูลที่มาของโมเดล (เสียงจริง / ชนิดช้าง / แหล่งข้อมูล / ผลทดสอบ) อยู่ใน metadata ของไฟล์ ONNX และ `model_info` ใน `.pt`
สร้างใหม่ได้ด้วย `python -m elephant_rumble.export_onnx`

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
  export_onnx.py แปลงโมเดลเป็น ONNX + ตรวจผลเทียบ PyTorch
  elp.py        ดาวน์โหลดเสียงช้างจริง (Cornell ELP) + แบ่ง 80/10/10 + ประเมินไฟล์ 24 ชม.
  demo_long.py  ประเมินระดับเหตุการณ์บนไฟล์ยาวจำลอง
models/sincnet_lstm_real_elephant_data.pt    โมเดลเสียงช้างจริง ข้อมูลจริง (ค่าเริ่มต้น)
models/sincnet_lstm_real_elephant_data.onnx  ตัวเดียวกันในรูปแบบ ONNX
models/sinc_lstm.pt      โมเดลเสียงจำลอง
```
