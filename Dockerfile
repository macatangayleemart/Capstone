# Use official Python 3.11 image
FROM python:3.11-slim

# Install essential system dependencies for OpenCV, TensorFlow, and YOLO
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    ffmpeg \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first (for better caching)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files
COPY . .

# Collect static files
RUN python manage.py collectstatic --noinput || true

# EXPOSE is optional; Railway injects the port automatically
# EXPOSE 8000  <-- REMOVE or comment out

# Run migrations and start server using runtime PORT
CMD sh -c "python manage.py migrate && gunicorn myCapstone.wsgi:application --bind 0.0.0.0:$PORT"
