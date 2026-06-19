FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY tests/ tests/

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "src.main:server", "--host", "127.0.0.1", "--port", "8000"]
