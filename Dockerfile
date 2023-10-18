FROM python:3.9.5

ENV PYTHONUNBUFFERED=1

WORKDIR /usr/app/src/TCM

RUN pip3 install numpy==1.23.1 scipy==1.11 pynn==0.11.0
RUN pip3 install NEURON==8.0
RUN pip3 install nrnutils==0.3.0
RUN pip3 install pyyaml

COPY ./*.mod ./
COPY ./*.hoc ./

RUN nrnivmodl

COPY ./*.py ./

ENTRYPOINT [ "python3", "/usr/app/src/TCM/TCM_Build_Run.py"]