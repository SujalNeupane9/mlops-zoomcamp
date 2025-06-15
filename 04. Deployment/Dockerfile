# Use the provided base image that already contains the models
FROM agrigorev/zoomcamp-model:mlops-2024-3.10.13-slim

# Set working directory
WORKDIR /app

# Copy your local inference script into the container
COPY predict.py .

# Run the inference script by default
CMD ["python", "predict.py"]
