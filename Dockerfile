FROM python:3.11

# basic packages
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
        curl \
        htop \
        locales \
        tree \
        tzdata \
        gcc \
        git

# time zone and languages
ENV TZ=Asia/Taipei \
    LC_ALL=en_US.UTF-8 \
    LANG=en_US.UTF-8 \
    LANGUAGE=en_US.UTF-8

RUN locale-gen en_US.UTF-8 \
 && echo $TZ | tee /etc/timezone \
 && dpkg-reconfigure --frontend noninteractive tzdata

WORKDIR /rafael
COPY . /rafael

# prepare for the services
RUN mkdir /rafael/services/configs
RUN mkdir /rafael/services/log
RUN mkdir /rafael/services/results

# uvicorn parameters
ENV ROLE="server"
ENV PORT=8000
ENV PING_INTERVAL=600
ENV PING_TIMEOUT=300
ENV MAX_MESSAGE_SIZE=1e20

# configuration parameters
ENV SERVER_NODE_ID=""
ENV COMPENSATOR_NODE_ID=""
ENV CLIENT_NODE_ID=""
ENV SERVER_LOG_PATH=""
ENV COMPENSATOR_LOG_PATH=""
ENV CLIENT_LOG_PATH=""
ENV PROTOCOL=""
ENV SERVER_HOST=""
ENV SERVER_PORT=""
ENV COMPENSATOR_HOST=""
ENV COMPENSATOR_PORT=""

RUN pip install poetry \
 && poetry env use python3 \
 && poetry install

CMD ["/bin/sh", "-c", "poetry run python3 services/service.py ${ROLE} --host 0.0.0.0 --port ${PORT} --ws-ping-interval ${PING_INTERVAL} --ws-ping-timeout ${PING_TIMEOUT} --max-message-size ${MAX_MESSAGE_SIZE}"]
