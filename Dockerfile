FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py dashboard.html ./
EXPOSE 5050
HEALTHCHECK CMD python -c "import urllib.request;urllib.request.urlopen('http://localhost:5050/health')"
CMD ["python", "app.py"]
