FROM python:3.10.14

WORKDIR /tg-siri

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY gpt4o-mini.py tg_bot.py utils.py .env /tg-siri/

CMD ["python", "tg_bot.py"]

