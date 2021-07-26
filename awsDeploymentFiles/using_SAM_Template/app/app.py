try:
    import torch
    import torchvision
    import torch.nn as nn

    import base64
    import json
    import io
    import boto3
    import requests
    import zipfile
    import io
    import json
    import shutil
    from pathlib import Path

    import numpy as np
    import pandas as pd
    import torch.nn.functional as F

    from PIL import Image
    from io import BytesIO
    from collections import OrderedDict

    import seaborn as sns
    from matplotlib import pyplot as plt

    sns.set()
    sns.set_style("whitegrid")
    sns.set_style("ticks", {"xtick.major.size": 8, "ytick.major.size": 8})
    sns.axes_style("whitegrid")
except Exception as e:
    print('### Exception occured while importing modules: {}'.format(str(e)))

class IplPrediction(object):
    seed = 143
    torch.manual_seed(seed)
    
    s3_obj = boto3.client('s3')
    res_dataset = s3_obj.get_object(Bucket = 'suman-projects', Key='IPL_model_trained/all_matches.csv')
    
    def __init__(self, epochs=40, lr=0.1, split_ratio=(70.0, 30.0), train_bs=32, pltName='v2_test15.png'):
        self.epochs = epochs
        self.lr = lr
        self.split_ratio = split_ratio
        self.train_bs = train_bs   # Then test_bs = 2*train_bs
        self.pltName = pltName
        self.modelPTH = self.pltName.split('.')[0]+'_model.pt'

        self.dataset_path = Path(Path.cwd(), 'dataset')
        self.snippets_path = Path(Path.cwd(), 'snippets')
        self.filename = 'all_matches.csv'
        self.recordedMetrics = 'Recorded_Metrics.csv'

        self.json_path = Path.joinpath(self.dataset_path, 'label_encode.json')

    # Step-1 : Load Dataset
    def loadDataSet(self):
        data = io.BytesIO(res_dataset['Body'].read())
        self.df_parsed = pd.read_csv(data)
    
    # Step-2: Clean Dataset
    def cleanDataSet(self):
        self.df_parsed.drop(columns=['wides', 'noballs', 'byes', 'legbyes', 'penalty', 'wicket_type',
                            'other_wicket_type', 'other_player_dismissed', 'season', 'start_date'], axis=1, inplace=True)
        non_exist_teams = ['Kochi Tuskers Kerala',
                           'Pune Warriors',
                           'Rising Pune Supergiants',
                           'Rising Pune Supergiant',
                           'Gujarat Lions']
        mask_bat_team = self.df_parsed['batting_team'].isin(non_exist_teams)
        mask_bow_team = self.df_parsed['bowling_team'].isin(non_exist_teams)
        self.df_parsed = self.df_parsed[~mask_bat_team]
        self.df_parsed = self.df_parsed[~mask_bow_team]


##########################################
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


def encode_nd_normalize_input(input, encode_data=None, mms_data=None):
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
        # print('### dir of boto3 : {}'.format(dir(s3)))

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
        event['body']['input_str'], encode_data, mms_data)

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


def ipl_EDA(bat, bowl):
    ds_path = '/opt/ml/all_matches.csv'
    df = pd.read_csv(ds_path, usecols=['match_id', 'venue', 'innings',
                     'ball', 'batting_team', 'bowling_team', 'runs_off_bat', 'extras'])

    # Clean Dataset :
    # 1. Drop columns
    #df.drop(columns=['season', 'start_date', 'striker', 'non_striker', 'bowler', 'wides', 'noballs', 'byes', 'legbyes','penalty', 'wicket_type', 'player_dismissed', 'other_wicket_type', 'other_player_dismissed'], inplace=True)

    # 2. Delete Non-existing teams : 'Kochi Tuskers Kerala' 'Pune Warriors','Rising Pune Supergiants', 'Rising Pune Supergiant','Gujarat Lions'
    mask_bat_team = df['batting_team'].isin(['Kochi Tuskers Kerala',
                                             'Pune Warriors',
                                             'Rising Pune Supergiants',
                                             'Rising Pune Supergiant',
                                             'Gujarat Lions'
                                             ])
    mask_bow_team = df['bowling_team'].isin(['Kochi Tuskers Kerala',
                                             'Pune Warriors',
                                             'Rising Pune Supergiants',
                                             'Rising Pune Supergiant',
                                             'Gujarat Lions'
                                             ])
    df = df[~mask_bat_team]
    df = df[~mask_bow_team]

    # 3. Replace the old team names with new team name:
    df.loc[df.batting_team == 'Delhi Daredevils',
           'batting_team'] = 'Delhi Capitals'
    df.loc[df.batting_team == 'Deccan Chargers',
           'batting_team'] = 'Sunrisers Hyderabad'

    df.loc[df.bowling_team == 'Delhi Daredevils',
           'bowling_team'] = 'Delhi Capitals'
    df.loc[df.bowling_team == 'Deccan Chargers',
           'bowling_team'] = 'Sunrisers Hyderabad'

    # 4. Replace venue column unique names :
    df.loc[df.venue == 'M.Chinnaswamy Stadium',
           'venue'] = 'M Chinnaswamy Stadium'
    df.loc[df.venue == 'Punjab Cricket Association IS Bindra Stadium, Mohali',
           'venue'] = 'Punjab Cricket Association Stadium'
    df.loc[df.venue == 'Punjab Cricket Association IS Bindra Stadium',
           'venue'] = 'Punjab Cricket Association Stadium'
    df.loc[df.venue == 'Wankhede Stadium, Mumbai', 'venue'] = 'Wankhede Stadium'
    df.loc[df.venue == 'Rajiv Gandhi International Stadium, Uppal',
           'venue'] = 'Rajiv Gandhi International Stadium'
    df.loc[df.venue == 'MA Chidambaram Stadium, Chepauk',
           'venue'] = 'MA Chidambaram Stadium'
    df.loc[df.venue == 'MA Chidambaram Stadium, Chepauk, Chennai',
           'venue'] = 'MA Chidambaram Stadium'

    # print('### Total {} : venue details present '.format(len(df.venue.unique())))
    # print('### Total {}  : Batting teams are there'.format(len(df.batting_team.unique())))
    # print('### Total {}  : Bowlling teams are there'.format(len(df.bowling_team.unique())))
    # print(df.shape)

    df_req = df.query(
        f'(batting_team == "{bat}" and bowling_team == "{bowl}") or (batting_team == "{bowl}" and bowling_team == "{bat}")').copy()
    df_req['finalScore'] = df_req['runs_off_bat'] + df_req['extras']

    total_matches_played = df_req.match_id.unique().shape[0]

    df_score = pd.concat([
        df_req.groupby(['match_id', 'venue', 'innings', 'batting_team', 'bowling_team'])[
            'finalScore'].sum(),
        df_req.loc[df.ball < 6.1].groupby(['match_id', 'venue', 'innings', 'batting_team', 'bowling_team'])['finalScore'].sum()],
        axis=1,
        keys=['20-score', '6-score']
    )

    # df_score.to_csv('/opt/tmp.csv')
    #df_plot = pd.read_csv('/opt/tmp.csv')
    # df_plot.head()
    df_plot = df_score.reset_index()
    win_history = {}
    win_series = []
    for k, v in df_score.groupby(['match_id']):
        # print('### {} : {} won'.format(k,v.loc[v['20-score']==v['20-score'].values.max()].index[0][3]))
        win_series.append(
            v.loc[v['20-score'] == v['20-score'].values.max()].index[0][3])
        win_series.append(None)

        win_history[k] = v.loc[v['20-score'] ==
                               v['20-score'].values.max()].index[0][3]

    # win_history
    sns.set_theme(style="darkgrid")
    print('### Total number of matches played with each other : {} = {} - won by {} team + {} - won by {} team'.format(
        total_matches_played, list(win_history.values()).count(bat), bat, list(win_history.values()).count(bowl), bowl))
    df_plot['won'] = pd.Series(win_series)
    #df_plot.to_csv('tmp.csv', index=False)
    # Plot the graphs:
    sns.countplot(x=df_plot['won'])
    plt.title('Teams winning History Against each other in : {} matches {} & {}'.format(
        total_matches_played, list(win_history.values()).count(bat), list(win_history.values()).count(bowl)))
    plt.savefig('/tmp/1.png')

    fig, ax = plt.subplots(3, 2, figsize=(15, 15))
    fig.suptitle('IPL Exploratory Data Analysis (EDA)')

    # sns.countplot(x=df_plot['won'],ax=ax[0,0]);
    #ax[0,0].set_title('Teams winning History Against each other in : {} matches'.format(total_matches_played));

    barplt = sns.barplot(x=df_plot['venue'], y=df_plot['20-score'],
                         ax=ax[0, 0], hue='batting_team', data=df_plot)
    loc, labels = plt.xticks()
    barplt.set_xticklabels(barplt.get_xticklabels(),
                           rotation=90,
                           horizontalalignment='right')
    ax[0, 0].set_title('Selected Teams 20-Overs score')

    barplt = sns.barplot(x=df_plot['venue'], y=df_plot['6-score'],
                         ax=ax[0, 1], hue='batting_team', data=df_plot)
    loc, labels = plt.xticks()
    barplt.set_xticklabels(barplt.get_xticklabels(),
                           rotation=90,
                           horizontalalignment='right')
    ax[0, 1].set_title('Selected Teams 6-Overs score')

    sns.boxplot(x=df_plot['batting_team'], y=df_plot['20-score'], ax=ax[1, 0])
    ax[1, 0].set_title('5-Point Summary 20-Overs ')

    sns.boxplot(x=df_plot['batting_team'], y=df_plot['6-score'], ax=ax[1, 1])
    ax[1, 1].set_title('5-Point Summary 6-Overs ')

    sns.violinplot(x=df_plot['batting_team'],
                   y=df_plot['20-score'], ax=ax[2, 0])
    sns.violinplot(x=df_plot['batting_team'],
                   y=df_plot['6-score'], ax=ax[2, 1])

    # sns.displot(x=df_plot['6-score'],ax=ax[1,0]);

    #plt.setp(ax, yticks=[])
    plt.tight_layout()
    fig.savefig('/tmp/2.png')

    sns.lmplot(x='6-score', y='20-score', data=df_plot, hue='batting_team')
    plt.title('Innings vs Score')
    plt.tight_layout()
    plt.savefig('/tmp/3.png')
    from glob import glob
    print('### All plots are saved in : {}'.format(glob('/tmp/*.png')))

    s3 = boto3.client('s3')
    bucket = 'suman-projects'
    for imgname in ['1.png', '2.png', '3.png']:
        file_name = '/tmp/' + imgname
        s3.upload_file(file_name, bucket, imgname, ExtraArgs={
                       'GrantRead': 'uri="http://acs.amazonaws.com/groups/global/AllUsers"'})

    '''
    img_list = []
    img_dict = {}
    s3 = boto3.client('s3')
    bucket = 'your-bucket-name'
    file_name = 'location-of-your-file'
    key_name = 'name-of-file-in-s3'
    s3.upload_file(file_name, bucket, key_name)
    for imgname in ['1.png', '2.png', '3.png']:
        img = Image.open(imgname)
        img = img.resize((224,224))
        output_buffer = io.BytesIO()
        img.save(output_buffer, format='PNG')
        base64_str = base64.b64encode(output_buffer.getvalue())
        base64_str = base64_str.decode('utf-8')
        img_dict['_'.join(imgname.split('.'))] = base64_str
        img_list.append(base64_str)
        #break
    print('### Length of img dict : {}'.format(len(img_dict)))
    #img_dict['images'] = img_list
    print('### Type of base64 is : {}'.format(type(base64_str)))
    response = {
        "statusCode": 200,
        'headers': {
            'Access-Control-Allow-Headers': 'Content-Type',
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'OPTIONS,POST,GET'
        },
        "body": True
    }

    return response
    '''


def lambda_handler(event, context):
    print('### Lambda function execution started Event body is : {}'.format(
        event['body']))

    select_lambda = event.get('path', None)
    if select_lambda is not None:
        print('### Selected Lambda is : {}'.format(select_lambda))

    if select_lambda == "/classify_digit":
        response = predict_MNIST_Image(event)
    elif select_lambda == "/predict_ipl_score":
        response = predict_IPL_score(event)
        try:
            input_bytes = event['body'].encode('utf-8')
        except Exception as e:
            input_bytes = event['body']

        # print('### input_bytes is : {}'.format(input_bytes['input_str']))
        ipl_EDA(bowl=input_bytes['input_str'][3],
                bat=input_bytes['input_str'][2])

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
    # print('### Lambda response is : {}'.format(response))
    return response
