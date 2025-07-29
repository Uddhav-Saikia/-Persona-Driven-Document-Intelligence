FROM python:3.10-slim

WORKDIR /app

# Install system dependencies for PyMuPDF and GLib
RUN apt-get update && apt-get install -y --no-install-recommends \
    libmupdf-dev libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# COPY app/requirements.txt ./requirements.txt
# RUN pip install --no-cache-dir -r requirements.txt
# Copy project files
COPY main.py . 
COPY app /app

# Run the main script
CMD ["python", "main.py"]
