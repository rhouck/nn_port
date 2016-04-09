FROM b.gcr.io/tensorflow/tensorflow:latest
MAINTAINER Ryan Houck <ryanchouck@gmail.com>

RUN sudo apt-get update

# pandas hdf5 dependencies
RUN sudo apt-get install -y libhdf5-serial-dev
RUN sudo apt-get install -y python-tables

# cvxpy dependencies
RUN sudo apt-get install -y libatlas-base-dev gfortran
RUN sudo apt-get install -y python-dev
RUN sudo apt-get install -y python-numpy python-scipy

RUN pip install --upgrade pip

COPY requirements.txt /
RUN pip install -r /requirements.txt

RUN mkdir /project
WORKDIR /project