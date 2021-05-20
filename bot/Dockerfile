FROM python:3.8

COPY requirements.txt .

RUN pip install -r requirements.txt

RUN git clone https://github.com/facebookresearch/detectron2 /detectron2_repo

RUN pip install -e /detectron2_repo

COPY . .

CMD python3 bot.py
