FROM python:3.9.5

RUN apt-get update

WORKDIR /usr/app/src/TCM

RUN pip3 install numpy==1.25 scipy==1.11 pynn==0.11.0
RUN pip3 install NEURON==8.0
RUN pip3 install nrnutils==0.2.0
RUN pip3 install pyyaml

COPY ./*.py ./
COPY ./*.mod ./
COPY ./*.hoc ./

RUN nrnivmodl

ENTRYPOINT [ "python3", "/usr/app/src/TCM/TCM_Build_Run.py"]