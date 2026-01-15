from fastapi import FastAPI, UploadFile, File
import tensorflow as tf
import numpy as np
from io import BytesIO
from PIL import Image

app = FastAPI()
model = tf.keras.models.load_model('casting_model_v1.h5')

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # 1. Đọc ảnh từ Client gửi lên
    contents = await file.read()
    img = Image.open(BytesIO(contents)).convert('RGB')
    
    # 2. Tiền xử lý ảnh
    img = img.resize((300, 300))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # 3. Dự đoán
    prediction = model.predict(img_array)
    label = "OK" if prediction[0][0] > 0.5 else "DEFECTIVE"
    
    # 4. Trả về JSON
    return {"result": label, "confidence": float(prediction[0][0])}

# Chạy server bằng lệnh: uvicorn main:app --reload