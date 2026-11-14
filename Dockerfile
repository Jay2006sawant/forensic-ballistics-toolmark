FROM python:3.10-slim

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py compare_core.py report_builder.py styles.css ./
ENV GRADIO_SERVER_NAME=0.0.0.0
ENV PORT=7860

EXPOSE 7860
CMD ["python", "app.py"]