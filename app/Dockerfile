# app/Dockerfile

# 1. Use a lightweight Python base image
FROM python:3.9-slim

# 2. Set working directory inside the container
WORKDIR /app

# 3. Copy necessary files from your project to the container
# We copy 'src' (code), 'app' (api), and 'models' (brain)
COPY src/ /src/
COPY app/ /app/
COPY models/ /models/

# 4. Install dependencies
# We install torch, fastapi, and others needed for inference
RUN pip install torch numpy pandas fastapi uvicorn pydantic scikit-learn

# 5. Expose the port so we can access the API
EXPOSE 8000

# 6. Set python path so Python can find your 'src' modules
ENV PYTHONPATH="${PYTHONPATH}:/src"

# 7. Run the API using Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]