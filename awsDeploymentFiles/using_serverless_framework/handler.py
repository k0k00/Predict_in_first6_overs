try:
    import unzip_requirements
    import json
    import io
    import sys
    import boto3
    import os
    import torch
    import numpy as np
    print('### Import end')
except Exception as e:
    print(str(e))
    print('### Exception occured while import')


try:
    # define env variables if there are not existing
    S3_BUCKET = 'suman-projects'
    MODEL_PATH = 'IPL_model_trained/iplscorePred-v1.pth'
    LABEL_ENCODE_PATH = 'IPL_model_trained/label_encode.json'
    MIN_MAX_NORM = 'IPL_model_trained/min_max_norm_df.json'
    s3 = boto3.client('s3')
    dataset = s3.get_object(
        Bucket=S3_BUCKET, Key='IPL_model_trained/all_matches.csv')
    model_obj = s3.get_object(Bucket=S3_BUCKET, Key=MODEL_PATH)
    label_obj = s3.get_object(Bucket=S3_BUCKET, Key=LABEL_ENCODE_PATH)
    minmax_obj = s3.get_object(Bucket=S3_BUCKET, Key=MIN_MAX_NORM)
except Exception as e:
    print(str(e))
    print('### Exception occured while creating BOTO3 objects')

bytestream = io.BytesIO(model_obj['Body'].read())
print(sys.getsizeof(bytestream) // (1024 * 1024))
model = torch.load(bytestream)

encode_data = json.load(label_obj['Body'])
mms_data = json.load(minmax_obj['Body'])


class linearRegression(torch.nn.Module):
    def __init__(self, inputSize, outputSize):
        super(linearRegression, self).__init__()
        self.linear = torch.nn.Linear(inputSize, outputSize)

    def forward(self, x):
        out = self.linear(x)
        return out


loaded_model = linearRegression(20, 1)
loaded_model.load_state_dict(model)
loaded_model.eval()


def encode_nd_normalize_input(input):
    input_enc = {}
    columns = ['venue', 'innings', 'batting_team', 'bowling_team', 'bat1', 'bat2', 'bat3', 'bat4',
               'bat5', 'bat6', 'bat7', 'bat8', 'bat9', 'bat10', 'bow1', 'bow2', 'bow3', 'bow4', 'bow5', 'bow6']
    input = list(map(lambda x: x.strip(), input))
    input_dict = dict(zip(columns, input))
    # Label Encode input
    for col in input_dict:
        if col == 'venue' or col == 'batting_team' or col == 'bowling_team':
            if input_dict[col] in encode_data[col].keys():
                input_enc[col] = encode_data[col][input_dict[col]]
            else:
                input_enc[col] = 0
        elif col == 'innings':
            input_enc[col] = input_dict[col]
        elif col in ['bat1', 'bat2', 'bat3', 'bat4', 'bat5', 'bat6', 'bat7', 'bat8', 'bat9', 'bat10', 'bow1', 'bow2', 'bow3', 'bow4', 'bow5', 'bow6']:
            if input_dict[col] in encode_data['Players'].keys():
                input_enc[col] = encode_data['Players'][input_dict[col]]
            else:
                input_enc[col] = 0
    print('### Input_dict is : {}'.format(input_dict))
    print('### Input_enc is : {}'.format(input_enc))
    # Normalize input :
    input_data_norm = {}
    for col in input_enc:
        try:
            input_data_norm[col] = round((
                input_enc[col] - mms_data[col]['min'])/(mms_data[col]['max'] - mms_data[col]['min']), 4)
        except Exception as e:
            input_data_norm[col] = 0.0
    print('### Input_data_norm is : {}'.format(input_data_norm))
    return input_data_norm


def predict(event, context):
    print('### You are in predict function')
    body = {
        "message": "Your function executed successfully!",
        "input": event
    }
    print('### Event captured by handler is : {}'.format(body['input']))

    input_data_norm = encode_nd_normalize_input(
        json.loads(event['body'])['input_str'])

    print('### Normalized value of input : {}'.format(input_data_norm))
    predicted_score = loaded_model(torch.Tensor(
        np.array(list(input_data_norm.values()), dtype=float)))
    predicted_score_unnorm = predicted_score.item(
    )*(mms_data['Total_score']['max'] - mms_data['Total_score']['min'])+mms_data['Total_score']['min']

    response = {
        "statusCode": 200,
        "headers": {
            'Content-Type': 'application/json',
            'Access-Control-Allow-Origin': '*',
            "Access-Control-Allow-Credentials": True

        },
        "body": predicted_score_unnorm
    }

    return response
