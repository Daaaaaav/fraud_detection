FROM python:3.10

WORKDIR /app

RUN apt-get update && apt-get install -y \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt . 

RUN pip install --upgrade pip
RUN pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2
RUN pip install -r requirements.txt

COPY . .

EXPOSE 5001

CMD ["gunicorn", "app:app"]
