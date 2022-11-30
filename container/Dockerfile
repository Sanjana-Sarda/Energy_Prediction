FROM python:3.8.14

COPY requirements.txt ./

RUN pip install --no-cache-dir --upgrade pip \
  && pip install --no-cache-dir -r requirements.txt

RUN curl -LO https://dl.k8s.io/release/v1.21.5/bin/linux/amd64/kubectl \
  && mv kubectl /usr/local/bin

RUN git clone https://github.com/Sanjana-Sarda/Energy_Prediction \
  && cd Energy_Prediction 