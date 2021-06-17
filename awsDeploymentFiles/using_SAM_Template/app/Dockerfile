# FROM public.ecr.aws/lambda/python:3.8
FROM 936131757702.dkr.ecr.ap-south-1.amazonaws.com/torchbaseimage:latest

COPY app.py requirements.txt ./
COPY model /opt/ml/model

RUN python3.8 -m pip install -r requirements.txt -t .

CMD ["app.lambda_handler"]
