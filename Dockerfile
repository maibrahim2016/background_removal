FROM jenkins:latest
USER root
RUN mkdir /my_app
WORKDIR /my_app
COPY requirement.txt /my_app
RUN pwd
RUN ls -la
RUN apr-get update
RUN apt-get install -y python-pip
