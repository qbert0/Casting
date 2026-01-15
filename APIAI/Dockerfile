FROM python:3.10-slim

WORKDIR /app

# Cài đặt thư viện hệ thống cho OpenCV và ảnh
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Lưu ý: Chúng ta sẽ không COPY model vào Image nếu dùng Docker Compose Volume
# Nhưng để chắc chắn chạy được mọi nơi, ta vẫn COPY code
COPY main.py .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]