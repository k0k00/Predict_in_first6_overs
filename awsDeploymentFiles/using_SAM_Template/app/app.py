try:
    import torch
    import torch.nn as nn
    import base64
    import json
    import io
    import boto3
    import io
    import json
    from pathlib import Path

    import numpy as np
    import pandas as pd
    import torch.nn.functional as F

    from PIL import Image

    import seaborn as sns
    from matplotlib import pyplot as plt

    sns.set()
    sns.set_style("whitegrid")
    sns.set_style("ticks", {"xtick.major.size": 8, "ytick.major.size": 8})
    sns.axes_style("whitegrid")
except Exception as e:
    print('### Exception occured while importing modules: {}'.format(str(e)))

# S3 operations:


class S3_Utils(object):

    s3_obj = boto3.client('s3')
    S3_BUCKET = 'suman-projects'

    def __init__(self, MODEL_PATH=None, LABEL_ENCODE_PATH=None, **kwargs):
        self.MODEL_PATH = MODEL_PATH
        self.LABEL_ENCODE_PATH = LABEL_ENCODE_PATH

    @classmethod
    def get_s3_object(cls, key_path):
        return cls.s3_obj.get_object(Bucket=cls.S3_BUCKET, Key=key_path)
    
    @classmethod
    def upload_file_to_s3(cls,local_file_path,remote_filename):
        #print('### Uploading local file : {} with name remote file name : {}'.format(local_file_path,remote_filename))
        cls.s3_obj.upload_file(local_file_path,cls.S3_BUCKET,remote_filename,ExtraArgs={
                       'GrantRead': 'uri="http://acs.amazonaws.com/groups/global/AllUsers"'})

# Model selevction:
class linearRegression(torch.nn.Module):
    def __init__(self, inputSize, outputSize):
        super(linearRegression, self).__init__()
        self.linear = torch.nn.Linear(inputSize, outputSize)

    def forward(self, x):
        out = self.linear(x)
        return out

# IPL Score Prediction


class Predict_IplPowerPlayScore(object):
    def __init__(self, event):
        self.event = event

    def __repr__(self):
        return str('### Event is : \n\t{}'.format(self.event))

    # 1. Parse Input Event
    def parse_event_body(self):
        input_bytes = self.event['body'].encode('utf-8')
        try:
            self.even['body'] = json.loads(base64.b64decode(input_bytes))
        except Exception as e:
            self.event['body'] = json.loads(input_bytes)

    # 2. Prepare Input Data:
    def get_input_data(self):
        columns = ['venue', 'innings', 'batting_team', 'bowling_team', 'bat1', 'bat2', 'bat3', 'bat4',
                   'bat5', 'bat6', 'bat7', 'bat8', 'bat9', 'bat10', 'bow1', 'bow2', 'bow3', 'bow4', 'bow5', 'bow6']
        input_str = self.event['body']['input_str']
        input_data = dict(zip(columns, input_str))
        return input_data

    # 3. Encode Input Features
    def get_input_encode_data(self, input_data, encode_data):
        input_encoded_data = {}
        for col in input_data:
            try:
                if col not in ['innings', 'bat1', 'bat2', 'bat3', 'bat4', 'bat5', 'bat6', 'bat7', 'bat8', 'bat9', 'bat10', 'bow1', 'bow2', 'bow3', 'bow4', 'bow5', 'bow6']:
                    input_encoded_data[col] = encode_data[col][input_data[col]]
                elif col == 'innings':
                    input_encoded_data[col] = float(input_data[col].strip())
                elif col in ['bat1', 'bat2', 'bat3', 'bat4', 'bat5', 'bat6', 'bat7', 'bat8', 'bat9', 'bat10', 'bow1', 'bow2', 'bow3', 'bow4', 'bow5', 'bow6']:
                    input_encoded_data[col] = encode_data['Players'][input_data[col]
                                                                     ] if input_data[col] != "None" else 0
            except Exception as e:
                print('### Exception occured while processing column : {}'.format(col))
        return input_encoded_data

    # 4. Normalize Encoded values
    def get_input_norm_data(self, input_encoded, encode_data):
        input_norm_data = {}
        for col in input_encoded:
            try:
                if col not in ['innings', 'bat1', 'bat2', 'bat3', 'bat4', 'bat5', 'bat6', 'bat7', 'bat8', 'bat9', 'bat10', 'bow1', 'bow2', 'bow3', 'bow4', 'bow5', 'bow6']:
                    input_norm_data[col] = (input_encoded[col] - min(encode_data[col].values())) / (
                        max(encode_data[col].values()) - min(encode_data[col].values()))
                elif col in ['bat1', 'bat2', 'bat3', 'bat4', 'bat5', 'bat6', 'bat7', 'bat8', 'bat9', 'bat10', 'bow1', 'bow2', 'bow3', 'bow4', 'bow5', 'bow6']:
                    input_norm_data[col] = (input_encoded[col] - min(encode_data['Players'].values())) / (
                        max(encode_data['Players'].values()) - min(encode_data['Players'].values()))
                elif col == 'innings':
                    input_norm_data[col] = (input_encoded[col] - 1)

            except Exception as e:
                print(str(e), input_encoded[col])

        return input_norm_data

    # 5. IPL EDA Graphs :
    def ipl_EDA(self, file_obj, **kwargs):
        bat=self.event['body']['input_str'][2]
        bowl=self.event['body']['input_str'][3]
        print(bat,bowl,file_obj)
        import pandas as pd
        df = pd.read_csv(file_obj, usecols=['match_id', 'venue', 'innings','ball', 'batting_team', 'bowling_team', 'runs_off_bat', 'extras'])

        # 1. Delete Non-existing teams :
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

        # 2. Replace the old team names with new team name:
        df.loc[df.batting_team == 'Delhi Daredevils',
               'batting_team'] = 'Delhi Capitals'
        df.loc[df.batting_team == 'Deccan Chargers',
               'batting_team'] = 'Sunrisers Hyderabad'

        df.loc[df.bowling_team == 'Delhi Daredevils',
               'bowling_team'] = 'Delhi Capitals'
        df.loc[df.bowling_team == 'Deccan Chargers',
               'bowling_team'] = 'Sunrisers Hyderabad'

        # 3. Replace venue column unique names :
        df.loc[df.venue == 'M.Chinnaswamy Stadium',
               'venue'] = 'M Chinnaswamy Stadium'
        df.loc[df.venue == 'Punjab Cricket Association IS Bindra Stadium, Mohali',
               'venue'] = 'Punjab Cricket Association Stadium'
        df.loc[df.venue == 'Punjab Cricket Association IS Bindra Stadium',
               'venue'] = 'Punjab Cricket Association Stadium'
        df.loc[df.venue == 'Wankhede Stadium, Mumbai',
               'venue'] = 'Wankhede Stadium'
        df.loc[df.venue == 'Rajiv Gandhi International Stadium, Uppal',
               'venue'] = 'Rajiv Gandhi International Stadium'
        df.loc[df.venue == 'MA Chidambaram Stadium, Chepauk',
               'venue'] = 'MA Chidambaram Stadium'
        df.loc[df.venue == 'MA Chidambaram Stadium, Chepauk, Chennai',
               'venue'] = 'MA Chidambaram Stadium'

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
        import seaborn as sns
        from matplotlib import pyplot as plt
        
        sns.set_theme(style="darkgrid")
        print('### Total number of matches played with each other : {} = {} - won by {} team + {} - won by {} team'.format(
            total_matches_played, list(win_history.values()).count(bat), bat, list(win_history.values()).count(bowl), bowl))
        df_plot['won'] = pd.Series(win_series)

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
        
        for imgname in ['1.png', '2.png', '3.png']:
            file_name = '/tmp/' + imgname
            S3_Utils.upload_file_to_s3(local_file_path=file_name,remote_filename=imgname)



# Lambda Event Process:
def lambda_handler(event, context):
    select_lambda = event.get('path', None)
    print(f'### Selected Lambda path : {select_lambda}')

    if select_lambda is not None and select_lambda == "/predict_ipl_score":
        ipl_obj = Predict_IplPowerPlayScore(event)
        #print(ipl_obj)
        # 1. Convert and update event body as a dict
        ipl_obj.parse_event_body()

        # 2. Prepare model input = dict(zip(columns,input_str))
        input_data = ipl_obj.get_input_data()

        # 3. S3: Load encode json file from S3:
        encode_obj = S3_Utils.get_s3_object(
            key_path='IPL_model_trained/label_encode.json')

        encode_data = json.load(encode_obj['Body'])

        # 4. Encode input data with encode data :
        input_encoded_data = ipl_obj.get_input_encode_data(
            input_data, encode_data)

        # 5. Normalize Data:
        input_norm_data = ipl_obj.get_input_norm_data(
            input_encoded=input_encoded_data, encode_data=encode_data)

        # 6. S3: Load Model.pt parameters file from S3:
        model_obj = S3_Utils.get_s3_object(
            key_path='IPL_model_trained/v2_test15_model.pt')
        bytestream = io.BytesIO(model_obj['Body'].read())
        model_params = torch.load(bytestream)

        # 7. Select and Load Model:
        model = linearRegression(20, 1)
        model.load_state_dict(model_params['model_state_dict'])
        model.eval()
        print('### Predict IPL model is loaded : {}'.format(model.eval()))

        # 8. Prediction Logic:
        test_inp = torch.Tensor(list(input_norm_data.values()))
        pred = (model(torch.Tensor(test_inp)) *
                (encode_data['Total_score_max'] - encode_data['Total_score_min'])) + encode_data['Total_score_min']

        predicted_score = round(pred.item())

        # 9. EDA Graphs:
        dataset_obj = S3_Utils.get_s3_object(
            key_path='IPL_model_trained/all_matches.csv')
        bytestream = io.BytesIO(dataset_obj['Body'].read())
        ipl_obj.ipl_EDA(file_obj=bytestream)

        response = {
            "statusCode": 200,
            'headers': {
                'Access-Control-Allow-Headers': 'Content-Type',
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Methods': 'OPTIONS,POST,GET'
            },
            "body": predicted_score
        }

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
