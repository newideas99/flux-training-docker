FROM navinhariharan/fluxd-model:latest

WORKDIR /
RUN git clone https://github.com/ostris/ai-toolkit.git

WORKDIR /ai-toolkit

RUN git submodule update --init --recursive
RUN pip install torch runpod
RUN pip install -r requirements.txt

ENV HF_HOME=/huggingface/

COPY main.py /ai-toolkit/runpod_serverless.py
COPY test_input.json /ai-toolkit/test_input.json
COPY dataset_downloader.py /ai-toolkit/dataset_downloader.py
COPY training-config-dev.yaml /ai-toolkit/config/training-config-dev.yaml

CMD ["python", "-u", "runpod_serverless.py"]
