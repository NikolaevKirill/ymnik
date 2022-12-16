FROM python:3.8.5

RUN mkdir -p /usr/app/
WORKDIR /usr/app/

COPY . /usr/app/

RUN pip install -r requirements.txt

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]