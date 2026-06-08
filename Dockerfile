
FROM python:3.12


WORKDIR /code


COPY ./requirements.txt /code/requirements.txt


RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt


COPY ./app /code/app


# Use shell form for CMD so we can access the Render $PORT variable, defaulting to 80 locally
CMD fastapi run app/main.py --port ${PORT:-80}