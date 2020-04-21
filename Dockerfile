FROM microservice:latest
USER root
RUN mkdir /python_docker_jenkins
WORKDIR /python_docker_jenkins
COPY requirement.txt /python_docker_jenkins
RUN pwd
RUN ls -la
RUN apr-get update
RUN apt-get install -y python-pip
