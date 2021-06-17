try:
    import torch
    import torchvision
    import base64
    import json
    import io
    import boto3
    import numpy as np

    import torch.nn as nn
    import torch.nn.functional as F

    from PIL import Image
    from io import BytesIO

except Exception as e:
    print('### Exception occured while importing modules: {}'.format(str(e)))

#import ptvsd

#ptvsd.enable_attach(address=('0.0.0.0', 5890), redirect_output=True)
#ptvsd.wait_for_attach()

# Global variables:
image_transforms = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.1307,), (0.3081,))])


class Net_mnist(nn.Module):
    def __init__(self):
        super(Net_mnist, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, kernel_size=5)
        self.conv2 = nn.Conv2d(20, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()

        self.fc1 = nn.Linear(320, 100)
        self.bn1 = nn.BatchNorm1d(100)

        self.fc2 = nn.Linear(100, 100)
        self.bn2 = nn.BatchNorm1d(100)

        self.smax = nn.Linear(100, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))

        x = x.view(-1, 320)
        x = self.bn1(F.relu(self.fc1(x)))
        x = F.dropout(x, training=self.training)

        x = self.bn2(F.relu(self.fc2(x)))
        x = F.dropout(x, training=self.training)

        return F.softmax(self.smax(x), dim=-1)


class Net_IPL(torch.nn.Module):
    def __init__(self, inputSize, outputSize):
        super(Net_IPL, self).__init__()
        self.linear = torch.nn.Linear(inputSize, outputSize)

    def forward(self, x):
        out = self.linear(x)
        return out


def predict_MNIST_Image(event):
    model_file = '/opt/ml/model'
    model = Net_mnist()
    model.load_state_dict(torch.load(model_file))
    model.eval()

    image_bytes = event['body'].encode('utf-8')
    print('### image bytes : {}'.format(image_bytes))
    image = Image.open(
        BytesIO(base64.b64decode(image_bytes))).convert(mode='L')
    image = image.resize((28, 28))

    probabilities = model.forward(image_transforms(
        np.array(image)).reshape(-1, 1, 28, 28))
    label = torch.argmax(probabilities).item()

    return {
        'statusCode': 200,
        'headers': {
            'Access-Control-Allow-Headers': 'Content-Type',
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'OPTIONS,POST,GET'
        },
        'body': json.dumps(
            {
                "predicted_label": label,
            }
        )
    }


def encode_nd_normalize_input(input,encode_data=None,mms_data=None):
    print('### U r in encode norm function')
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


def predict_IPL_score(event):
    print('### You are in Predict IPL Lambda')
    input_bytes = event['body'].encode('utf-8')
    print(input_bytes)
    try:
        event['body'] = json.loads(base64.b64decode(input_bytes))
    except Exception as e:
        event['body'] = json.loads(input_bytes)
        
    print('### Event body is : {}'.format(event['body']['input_str']))

    try:
        S3_BUCKET = 'suman-projects'
        MODEL_PATH = 'IPL_model_trained/iplscorePred-v1.pth'
        LABEL_ENCODE_PATH = 'IPL_model_trained/label_encode.json'
        MIN_MAX_NORM = 'IPL_model_trained/min_max_norm_df.json'

        s3 = boto3.client('s3')
        #print('### dir of boto3 : {}'.format(dir(s3)))

        dataset = s3.get_object(
            Bucket=S3_BUCKET,
            Key='IPL_model_trained/all_matches.csv')
        model_obj = s3.get_object(Bucket=S3_BUCKET, Key=MODEL_PATH)
        label_obj = s3.get_object(Bucket=S3_BUCKET, Key=LABEL_ENCODE_PATH)
        minmax_obj = s3.get_object(Bucket=S3_BUCKET, Key=MIN_MAX_NORM)

        bytestream = io.BytesIO(model_obj['Body'].read())

        model = torch.load(bytestream)
        encode_data = json.load(label_obj['Body'])
        print('### encode_data Keys : {}'.format(encode_data.keys()))
        mms_data = json.load(minmax_obj['Body'])
        print('### mms_data Keys : {}'.format(mms_data.keys()))

    except Exception as e:
        print('### Exception occured while creating BOTO3 objects : {}'.format(str(e)))

    loaded_model = Net_IPL(20, 1)
    loaded_model.load_state_dict(model)
    loaded_model.eval()
    print('### Predict IPL model is loaded : {}'.format(loaded_model.eval()))

    input_data_norm = encode_nd_normalize_input(
            event['body']['input_str'],encode_data,mms_data)

    predicted_score = loaded_model(torch.Tensor(
        np.array(list(input_data_norm.values()), dtype=float)))

    predicted_score_unnorm = predicted_score.item(
    )*(mms_data['Total_score']['max'] - mms_data['Total_score']['min'])+mms_data['Total_score']['min']

    response = {
        "statusCode": 200,
        'headers': {
            'Access-Control-Allow-Headers': 'Content-Type',
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'OPTIONS,POST,GET'
        },
        "body": predicted_score_unnorm
    }

    return response


def lambda_handler(event, context):
    print('### Lambda function execution started')
    print('### Event body is : {}'.format(event['body']))
    select_lambda = event.get('path', None)
    if select_lambda is not None:
        print('### Selected Lambda is : {}'.format(select_lambda))

    if select_lambda == "/classify_digit":
        response = predict_MNIST_Image(event)
    elif select_lambda == "/predict_ipl_score":
        response = predict_IPL_score(event)
    else:
        response = {
            'statusCode': 403,
            'headers': {
                'Access-Control-Allow-Headers': 'Content-Type',
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Methods': 'OPTIONS,POST,GET'
            },
            'body': json.dumps(
                {
                    event['path']: 'Lambda Triggered, but given Lambda endpoint does not Exists',
                }
            )
        }

    return response