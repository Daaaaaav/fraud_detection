FROM python:3.10

WORKDIR /app

RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --upgrade pip
RUN pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2

RUN grep -v detectron2 requirements.txt > temp-requirements.txt && pip install -r temp-requirements.txt

RUN pip install 'git+https://github.com/facebookresearch/detectron2.git@9604f5995cc628619f0e4fd913453b4d7d61db3f'

COPY . .

EXPOSE 5001
CMD ["gunicorn", "app:app"]
