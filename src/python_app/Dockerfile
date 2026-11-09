FROM python:3.11-slim

WORKDIR /app

# Install pinned dependencies first (better layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the analysis script
COPY run_model.py .

# The dataset is NOT copied into the image; it is mounted at runtime to /app/data
ENV DATA_DIR=/app/data

CMD ["python", "run_model.py"]
