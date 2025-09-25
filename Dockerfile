FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 \
PYTHONUNBUFFERED=1
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
HEALTHCHECK CMD python -c "import sys; import os; sys.exit(0)"
CMD ["python", "main.py"]
