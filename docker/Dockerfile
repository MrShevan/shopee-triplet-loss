FROM nvidia/cuda:10.2-cudnn7-devel

RUN apt-get update -y && \
    apt-get install -y screen
RUN apt-get install -y python3-pip python-dev build-essential

ENV LC_CTYPE=C.UTF-8

COPY ./requirements.txt $PROJECT_ROOT/
RUN pip3 install --no-cache-dir -r requirements.txt

COPY jupyter_notebook_config.py /root/.jupyter/

RUN pip3 install --no-cache-dir tensorflow-gpu torch torchvision

RUN jupyter contrib nbextension install --user && \
    jupyter nbextensions_configurator enable --user && \
    jupyter nbextension enable code_prettify/code_prettify && \
    jupyter nbextension enable codefolding/main && \
    jupyter nbextension enable ExecuteTime && \
    jupyter nbextension enable freeze/main && \
    jupyter nbextension enable toc2/main && \
    jupyter nbextension enable  execute_time/ExecuteTime && \
    python3.6 -m ipykernel.kernelspec

CMD ["/bin/bash"]