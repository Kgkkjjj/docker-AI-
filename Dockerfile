FROM ubuntu:22.04
RUN apt-get update && apt-get install -y build-essential git pkg-config libgtk-3-dev
COPY . /app
WORKDIR /app
RUN gcc main.c -o docker-ai `pkg-config --cflags --libs gtk+-3.0`
CMD ["/app/docker-ai"]
