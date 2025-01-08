# # syntax=docker/dockerfile:1
# FROM python:3.10-bullseye

# # Expose the required port
# EXPOSE 6969

# # Set up working directory
# WORKDIR /app

# # Install system dependencies, clean up cache to keep image size small
# RUN apt update && \
#     apt install -y -qq ffmpeg && \
#     apt clean && rm -rf /var/lib/apt/lists/*

# # Copy application files into the container
# COPY . .

# # Create a virtual environment in the app directory and install dependencies
# RUN python3 -m venv /app/.venv && \
#     . /app/.venv/bin/activate && \
#     pip install --no-cache-dir --upgrade pip && \
#     pip install --no-cache-dir python-ffmpeg && \
#     pip install --no-cache-dir torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121 && \
#     if [ -f "requirements.txt" ]; then pip install --no-cache-dir -r requirements.txt; fi

# # Define volumes for persistent storage
# VOLUME ["/app/logs/"]

# # Set environment variables if necessary
# ENV PATH="/app/.venv/bin:$PATH"

# # Run the app
# ENTRYPOINT ["python3"]
# CMD ["app.py"]



# Use Python 3.10 slim-bullseye as base image for smaller size
FROM python:3.10-slim-bullseye

# Allow statements and log messages to immediately appear in the logs
ENV PYTHONUNBUFFERED True

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy application files
COPY . .

# Create and activate virtual environment
RUN python -m venv /app/venv
ENV PATH="/app/venv/bin:$PATH"

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir python-ffmpeg && \
    pip install --no-cache-dir torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cpu && \
    if [ -f "requirements.txt" ]; then pip install --no-cache-dir -r requirements.txt; fi

# Cloud Run will set PORT environment variable
ENV PORT 8080

# Create necessary directories with appropriate permissions
RUN mkdir -p /app/logs /app/data /app/outputs && \
    chmod -R 777 /app/logs /app/data /app/outputs

# Run as non-root user for security
RUN useradd -m -s /bin/bash appuser && \
    chown -R appuser:appuser /app
USER appuser

# Command to run the application
CMD exec uvicorn server_db_v0:app --host 0.0.0.0 --port ${PORT}