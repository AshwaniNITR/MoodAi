FROM python:3.9-slim

WORKDIR /app

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Make sure we have the model file included
COPY face_model.h5 /app/face_model.h5

# Verify the model file exists (will fail build if not)
RUN test -f /app/face_model.h5 || exit 1

# Expose the port the app runs on
EXPOSE 5000

# Command to run the application
CMD ["python", "app.py"]