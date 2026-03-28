# Use Python 3.11 slim base
FROM python:3.11-slim

# Install system dependencies needed for face_recognition (dlib) and OpenCV
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libopenblas-dev \
    liblapack-dev \
    libx11-dev \
    libgtk-3-dev \
    libboost-python-dev \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first (better Docker layer caching)
COPY requirements.txt .

# Install Python dependencies
# face-recognition installs dlib which takes a few minutes to compile
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the app
COPY . .

# Create known_faces directory if not present
RUN mkdir -p known_faces

# Expose port
EXPOSE 5000

# Start with gunicorn (production WSGI server)
CMD gunicorn --bind 0.0.0.0:$PORT --timeout 120 app:app
