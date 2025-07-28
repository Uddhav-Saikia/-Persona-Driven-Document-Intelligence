FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libmupdf-dev \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY . /app

COPY app/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", "main.py"]