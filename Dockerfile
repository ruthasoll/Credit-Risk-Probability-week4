# Dockerfile

FROM python:3.9-slim

WORKDIR /app

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY models/ ./models/ 
# Note: models might not exist if we built locally, ideally we pull from MLflow or S3 
# But for this task, we assume local build or volume mount. 
# We'll just create the dir if it doesn't exist to avoid error
RUN mkdir -p models

# Copy tests just in case
COPY tests/ ./tests/

# Set Python Path
ENV PYTHONPATH=/app

# Expose port
EXPOSE 8000

# Command to run the API
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
