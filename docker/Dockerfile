FROM tensorflow/tensorflow:2.15.0-gpu

WORKDIR "/src"

COPY ../requirements.txt /requirements.txt

RUN pip install --user --upgrade pip
RUN pip install --no-cache-dir --user -r /requirements.txt

COPY ./src/*.py ./