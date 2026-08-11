FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download spaCy model
RUN python -m spacy download en_core_web_sm

# Copy source code
COPY src/ ./src/
COPY backend/ ./backend/
COPY frontend/ ./frontend/
COPY models/ ./models/
COPY .env.example .env

# Expose ports
EXPOSE 8000
EXPOSE 7860

# Default: start FastAPI
# Override CMD to start Gradio instead if needed
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
