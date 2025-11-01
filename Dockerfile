# pick a stable, common base
FROM python:3.11-slim

# make Python output unbuffered
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# install OS deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# copy ONLY requirements first
COPY requirements.txt .

# install python deps
RUN pip install --no-cache-dir -r requirements.txt

# copy the rest of the repo. this will bring in src/, including src/data/*.csv
COPY . .

# default command: run your pipeline
#    this assumes main.py reads src/data/train.csv and src/data/test.csv
#    and writes src/data/survival_predictions.csv
CMD ["python", "src/python/run.py"]