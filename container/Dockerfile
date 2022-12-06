FROM python:3.8.14

COPY requirements.txt ./
COPY tensorflow-2.8.0-cp38-cp38-linux_x86_64.whl ./tensorflow-2.8.0-cp38-cp38-linux_x86_64.whl

RUN pip install --no-cache-dir --upgrade pip \
  && pip install --ignore-installed --upgrade tensorflow-2.8.0-cp38-cp38-linux_x86_64.whl \
  && pip install --no-cache-dir -r requirements.txt

RUN curl -LO https://dl.k8s.io/release/v1.21.5/bin/linux/amd64/kubectl \
  && mv kubectl /usr/local/bin

COPY NN_test.h5 ./NN_test.h5
COPY inference.py ./inference.py
COPY mean_orig.csv ./mean_orig.csv
COPY var_orig.csv ./var_orig.csv
COPY test_data_9278.csv ./test_data.csv