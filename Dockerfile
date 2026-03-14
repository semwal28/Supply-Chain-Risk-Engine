# Use a slightly larger base that includes build tools
FROM python:3.12

WORKDIR /app

# Update pip first
RUN pip install --upgrade pip

COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]