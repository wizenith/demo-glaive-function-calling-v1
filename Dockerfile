# This is a hacky way to get the model working in this base image
FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime

# we reinstall python for some reason this works
RUN apt-get update && apt-get install -y software-properties-common
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y python3.8 python3-pip
# hack to make python3.8 the default python3
RUN rm /opt/conda/bin/python3 && ln -s /usr/bin/python3.8 /opt/conda/bin/python3
RUN rm /opt/conda/bin/pip3 && ln -s /usr/bin/pip3 /opt/conda/bin/pip3

WORKDIR /

# Install git
RUN apt-get update && apt-get install -y git

# Install python packages
ADD requirements.txt requirements.txt
RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install setuptools setuptools-rust
RUN python3 -m pip install -r requirements.txt
# important: install torch using the installed python pip module
RUN python3 -m pip install torch

# Add your model weight files 
# (in this case we have a python script)
ADD download.py .
RUN python3 download.py

ADD . .

EXPOSE 8000

CMD python3 -u app.py