FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml .
COPY src/ ./src/
COPY configs/ ./configs/

RUN pip install --no-cache-dir -e "."

ENV PYTHONPATH=/app/src

EXPOSE 8000

CMD ["uvicorn", "serving.app:app", "--host", "0.0.0.0", "--port", "8000"]