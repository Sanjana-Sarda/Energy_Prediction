FROM python:3.8.14

COPY requirements.txt ./

RUN pip install --no-cache-dir --upgrade pip \
  && pip install --no-cache-dir -r requirements.txt

RUN curl -LO https://dl.k8s.io/release/v1.21.5/bin/linux/amd64/kubectl \
  && mv kubectl /usr/local/bin

COPY NN_test.h5 ./NN_test.h5
COPY inference.py ./inference.py
COPY mean_orig.csv ./mean_orig.csv
COPY var_orig.csv ./var_orig.csv
COPY test_data_9160.csv ./test_data.csv