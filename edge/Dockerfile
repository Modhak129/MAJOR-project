FROM python:3.10-slim
WORKDIR /app
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY edge/ ./edge/
COPY config/ ./config/
COPY data/partition.py ./data/partition.py
CMD ["sh", "-c", "python data/partition.py && python edge/fl_client.py"]
