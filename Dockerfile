# This is a hacky way to get the model working in this base image
FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime

# we reinstall python, for some reason this works
RUN apt-get update && apt-get install -y software-properties-common
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    python3.8 \
    python3-pip \
    python3.8-dev \
    libpython3.8-dev
    # libpython 3.8 necessary for running on Banana

# make "python3" -> new installed python3.8
RUN rm /opt/conda/bin/python3 && ln -s /usr/bin/python3.8 /opt/conda/bin/python3
# note: "pip3" will still reference the base install, but we don't use it
# use this pattern at your own risk, Banana doesn't ensure future compatibility

WORKDIR /

# Install git
RUN apt-get update && apt-get install -y git

# Install python packages
ADD requirements.txt requirements.txt
# explicitly use "python3 -m pip" to avoid using pip3
RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install setuptools setuptools-rust
RUN python3 -m pip install -r requirements.txt
# reinstall torch in the pip module
RUN python3 -m pip install torch

# Add your model weight files 
# (in this case we have a python script)
ADD download.py .
RUN python3 download.py

ADD . .

EXPOSE 8000

CMD python3 -u app.py