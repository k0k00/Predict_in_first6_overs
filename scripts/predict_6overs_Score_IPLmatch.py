import requests
import zipfile
import io
import pdb
import shutil
import json
import time
from pathlib import Path
from datetime import date
from collections import OrderedDict

import pandas as pd
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import torch.nn.functional as F


from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler

dataset_path = Path.joinpath(Path.cwd().parent, 'dataset')
filename = 'all_matches.csv'
json_path = Path.joinpath(dataset_path, 'label_encode.json')


def getCsvFile(url="https://cricsheet.org/downloads/ipl_male_csv2.zip"):
    res = requests.get(url, stream=True)
    if res.status_code == 200:
        print('### Downloading the CSV file')
        z = zipfile.ZipFile(io.BytesIO(res.content))
        if filename in z.namelist():
            z.extract(filename, dataset_path)
            print('### Extracted %s file' % filename)
        else:
            print('### %s : File not found in ZIP Artifact' % filename)


def downloadDataset():
    if not dataset_path.exists():
        Path.mkdir(dataset_path)
        print('### Created Dataset folder')
        getCsvFile()
    # shutil.rmtree(dataset_path,ignore_errors=True)
    elif dataset_path.exists():
        files = [file for file in dataset_path.iterdir() if file.name ==
                 'all_matches.csv']
        if len(files) == 0:
            getCsvFile()


def parseDataset(csv_file=Path.joinpath(dataset_path, filename)):
    df = pd.read_csv(csv_file, parse_dates=['start_date'])
    print(df.batting_team.unique())
    print('### original dataframe shape is : {}'.format(df.shape))
    #df_parsed = df.loc[df['start_date'].dt.year < date.today().year]
    #df_2021 = df.loc[df['start_date'].dt.year >= date.today().year]

    df_parsed = df.copy(deep=True)
    # 1. Drop columns
    df_parsed.drop(columns=['season', 'start_date'], inplace=True)

    # 2. Delete Non-existing teams : 'Kochi Tuskers Kerala' 'Pune Warriors','Rising Pune Supergiants', 'Rising Pune Supergiant','Gujarat Lions'
    non_exist_teams = ['Kochi Tuskers Kerala',
                       'Pune Warriors',
                       'Rising Pune Supergiants',
                       'Rising Pune Supergiant',
                       'Gujarat Lions']
    mask_bat_team = df_parsed['batting_team'].isin(non_exist_teams)
    mask_bow_team = df_parsed['bowling_team'].isin(non_exist_teams)
    df_parsed = df_parsed[~mask_bat_team]
    df_parsed = df_parsed[~mask_bow_team]

    # 3. Replace the old team names with new team name:
    df_parsed.loc[df_parsed.batting_team ==
                  'Delhi Daredevils', 'batting_team'] = 'Delhi Capitals'
    df_parsed.loc[df_parsed.batting_team == 'Deccan Chargers',
                  'batting_team'] = 'Sunrisers Hyderabad'
    df_parsed.loc[df_parsed.batting_team ==
                  'Punjab Kings', 'batting_team'] = 'Kings XI Punjab'

    df_parsed.loc[df_parsed.bowling_team ==
                  'Delhi Daredevils', 'bowling_team'] = 'Delhi Capitals'
    df_parsed.loc[df_parsed.bowling_team == 'Deccan Chargers',
                  'bowling_team'] = 'Sunrisers Hyderabad'
    df_parsed.loc[df_parsed.bowling_team ==
                  'Punjab Kings', 'bowling_team'] = 'Kings XI Punjab'

    # 4. Replace venue column unique names :
    df_parsed.loc[df_parsed.venue == 'M.Chinnaswamy Stadium',
                  'venue'] = 'M Chinnaswamy Stadium'
    df_parsed.loc[df_parsed.venue == 'Brabourne Stadium, Mumbai',
                  'venue'] = 'Brabourne Stadium'
    df_parsed.loc[df_parsed.venue == 'Punjab Cricket Association IS Bindra Stadium, Mohali',
                  'venue'] = 'Punjab Cricket Association Stadium'
    df_parsed.loc[df_parsed.venue == 'Punjab Cricket Association IS Bindra Stadium',
                  'venue'] = 'Punjab Cricket Association Stadium'
    df_parsed.loc[df_parsed.venue == 'Wankhede Stadium, Mumbai',
                  'venue'] = 'Wankhede Stadium'
    df_parsed.loc[df_parsed.venue == 'Rajiv Gandhi International Stadium, Uppal',
                  'venue'] = 'Rajiv Gandhi International Stadium'
    df_parsed.loc[df_parsed.venue == 'MA Chidambaram Stadium, Chepauk',
                  'venue'] = 'MA Chidambaram Stadium'
    df_parsed.loc[df_parsed.venue == 'MA Chidambaram Stadium, Chepauk, Chennai',
                  'venue'] = 'MA Chidambaram Stadium'
    '''
    df_parsed.loc[df_parsed.batting_team ==
                  'Kolkata Knight Riders', 'batting_team'] = 'KKR'
    df_parsed.loc[df_parsed.batting_team ==
                  'Royal Challengers Bangalore', 'batting_team'] = 'RCB'
    df_parsed.loc[df_parsed.batting_team ==
                  'Chennai Super Kings', 'batting_team'] = 'CSK'
    df_parsed.loc[df_parsed.batting_team ==
                  'Kings XI Punjab', 'batting_team'] = 'KXIP'
    df_parsed.loc[df_parsed.batting_team ==
                  'Rajasthan Royals', 'batting_team'] = 'RR'
    df_parsed.loc[df_parsed.batting_team ==
                  'Delhi Capitals', 'batting_team'] = 'DC'
    df_parsed.loc[df_parsed.batting_team ==
                  'Sunrisers Hyderabad', 'batting_team'] = 'SRH'
    df_parsed.loc[df_parsed.batting_team ==
                  'Mumbai Indians', 'batting_team'] = 'MI'

    # Shorten the batting and bowling team names:
    df_parsed.loc[df_parsed.bowling_team ==
                  'Kolkata Knight Riders', 'bowling_team'] = 'KKR'
    df_parsed.loc[df_parsed.bowling_team ==
                  'Royal Challengers Bangalore', 'bowling_team'] = 'RCB'
    df_parsed.loc[df_parsed.bowling_team ==
                  'Chennai Super Kings', 'bowling_team'] = 'CSK'
    df_parsed.loc[df_parsed.bowling_team ==
                  'Kings XI Punjab', 'bowling_team'] = 'KXIP'
    df_parsed.loc[df_parsed.bowling_team ==
                  'Rajasthan Royals', 'bowling_team'] = 'RR'
    df_parsed.loc[df_parsed.bowling_team ==
                  'Delhi Capitals', 'bowling_team'] = 'DC'
    df_parsed.loc[df_parsed.bowling_team ==
                  'Sunrisers Hyderabad', 'bowling_team'] = 'SRH'
    df_parsed.loc[df_parsed.bowling_team ==
                  'Mumbai Indians', 'bowling_team'] = 'MI'
    '''

    # Rename striker and non-striker column names :
    df_parsed = df_parsed.rename(columns={
                                 'striker': 'batsmen', 'non_striker': 'batsmen_non_striker', 'bowler': 'bowlers'})
    # df.info()
    # Add total score column and filter data upto 6-overs
    df_parsed['Total_score'] = df_parsed.runs_off_bat + df_parsed.extras

    df_parsed.drop(columns=['wides', 'noballs', 'byes', 'legbyes', 'penalty', 'wicket_type',
                   'other_wicket_type', 'other_player_dismissed'], axis=1, inplace=True)

    df_parsed[(df_parsed.ball < 6.0) & (df_parsed.innings < 3)].to_csv(
        Path.joinpath(dataset_path, '1_parseDataset.csv'), index=False)

    print('### Total {} : venue details present '.format(
        len(df_parsed.venue.unique())))
    print('### Total {}  : Batting teams are there'.format(
        len(df_parsed.batting_team.unique())))
    print('### Total {}  : Bowlling teams are there'.format(
        len(df_parsed.bowling_team.unique())))
    print('### Batting teams are : {}'.format(df_parsed.batting_team.unique()))
    print('### Bowling teams are : {}'.format(df_parsed.bowling_team.unique()))
    print('### Shape of data frame after initial cleanup :{}'.format(df_parsed.shape))

    df_parsed = pd.read_csv(Path.joinpath(dataset_path, '1_parseDataset.csv'))
    # Path.joinpath(Path.cwd(),'1_parseDataset.csv').unlink()
    return df_parsed


def encodeLabels_save_to_file(df):
    label_encode_dict = {}
    players_df = pd.DataFrame(
        np.append(df.batsmen.unique(), df.bowlers.unique()), columns=['Players'])
    le = LabelEncoder()

    le.fit(players_df.Players)
    Players_e = le.transform(players_df.Players)
    Players_e_inv = le.inverse_transform(Players_e)
    label_encode_dict['Players'] = dict(
        zip(Players_e_inv, map(int, Players_e)))

    le.fit(df.batsmen)
    batsmen_e = le.transform(df.batsmen)
    batsmen_e_inv = le.inverse_transform(batsmen_e)
    label_encode_dict['batsmen'] = dict(
        zip(batsmen_e_inv, map(int, batsmen_e)))

    le.fit(df.bowlers)
    bowlers_e = le.transform(df.bowlers)
    bowlers_e_inv = le.inverse_transform(bowlers_e)
    label_encode_dict['bowlers'] = dict(
        zip(bowlers_e_inv, map(int, bowlers_e)))

    le.fit(df.venue)
    venue_e = le.transform(df.venue)
    venue_e_inv = le.inverse_transform(venue_e)
    label_encode_dict['venue'] = dict(zip(venue_e_inv, map(int, venue_e)))

    le.fit(df.batting_team)
    batting_team_e = le.transform(df.batting_team)
    batting_team_e_inv = le.inverse_transform(batting_team_e)
    label_encode_dict['batting_team'] = dict(
        zip(batting_team_e_inv, map(int, batting_team_e)))

    le.fit(df.bowling_team)
    bowling_team_e = le.transform(df.bowling_team)
    bowling_team_e_inv = le.inverse_transform(bowling_team_e)
    label_encode_dict['bowling_team'] = dict(
        zip(bowling_team_e_inv, map(int, bowling_team_e)))

    with open(json_path, 'w') as f:
        json.dump(label_encode_dict, f)


def format_data(df):
    print('### Shape of Dataframe before format_data : {}'.format(df.shape))
    Runs_off_Bat_6_overs = df.groupby(
        ['match_id', 'venue', 'innings', 'batting_team', 'bowling_team'])['runs_off_bat'].sum()
    Extras_6_overs = df.groupby(
        ['match_id', 'venue', 'innings', 'batting_team', 'bowling_team'])['extras'].sum()
    TotalScore_6_overs = df.groupby(
        ['match_id', 'venue', 'innings', 'batting_team', 'bowling_team'])['Total_score'].sum()

    Total_WktsDown = df.groupby(['match_id', 'venue', 'innings', 'batting_team', 'bowling_team'])[
        'player_dismissed'].count()

    bat_df = df.groupby(['match_id', 'venue', 'innings',
                        'batting_team', 'bowling_team'])['batsmen'].apply(list)
    bow_df = df.groupby(['match_id', 'venue', 'innings',
                        'batting_team', 'bowling_team'])['bowlers'].apply(list)

    pd.concat([bat_df, bow_df, Runs_off_Bat_6_overs, Extras_6_overs, TotalScore_6_overs, Total_WktsDown],
              axis=1).to_csv(Path.joinpath(dataset_path, '2_format_data.csv'))  # ,index=False)

    df = pd.read_csv(Path.joinpath(dataset_path, '2_format_data.csv'))
    # Path.joinpath(Path.cwd(),'2_format_data.csv').unlink()
    return df


def create_batsmen_bowler_df(df):
    bat = pd.DataFrame(np.zeros((df.shape[0], 10), dtype=float), columns=[
                       'bat1', 'bat2', 'bat3', 'bat4', 'bat5', 'bat6', 'bat7', 'bat8', 'bat9', 'bat10'])
    bowl = pd.DataFrame(np.zeros((df.shape[0], 6), dtype=float), columns=[
                        'bow1', 'bow2', 'bow3', 'bow4', 'bow5', 'bow6'])

    columns = ['bat1', 'bat2', 'bat3', 'bat4', 'bat5', 'bat6', 'bat7',
               'bat8', 'bat9', 'bat10', 'bow1', 'bow2', 'bow3', 'bow4', 'bow5', 'bow6']
    df_bat_bow = pd.concat([bat, bowl], axis=1)
    pd.concat([df, df_bat_bow], axis=1).to_csv(Path.joinpath(
        dataset_path, '3_create_batsmen_bowler_df.csv'), index=False)
    df = pd.read_csv(Path.joinpath(
        dataset_path, '3_create_batsmen_bowler_df.csv'))
    # Path.joinpath(dataset_path,'3_create_batsmen_bowler_df.csv').unlink()
    return df


def update_batsmen_bowler_column_names(df):
    for row, val in enumerate(df.batsmen):
        tmp = val[1:-1].split(', ')
        tmp_list = list(map(lambda x: x.strip("'"), tmp))
        tmp_list = list(OrderedDict.fromkeys(tmp_list))
        for i, j in enumerate(list(map(lambda x: x.strip("'"), tmp_list))):
            col = "bat%i" % (i+1)
            df[col][row] = j

    for row, val in enumerate(df.bowlers):
        tmp = val[1:-1].split(', ')
        tmp_list = list(map(lambda x: x.strip("'"), tmp))
        tmp_list = list(OrderedDict.fromkeys(tmp_list))
        for i, j in enumerate(list(map(lambda x: x.strip("'"), tmp_list))):
            col = "bow%i" % (i+1)
            df[col][row] = j

    df.to_csv(Path.joinpath(dataset_path,
              '4_update_batsmen_bowler_column_names.csv'), index=False)
    df = pd.read_csv(Path.joinpath(
        dataset_path, '4_update_batsmen_bowler_column_names.csv'))
    # Path.joinpath(dataset_path,'4_update_batsmen_bowler_column_names.csv').unlink()
    return df


def model_df(df):
    df_model = df[['venue', 'innings', 'batting_team', 'bowling_team', 'bat1', 'bat2', 'bat3', 'bat4', 'bat5', 'bat6', 'bat7', 'bat8','bat9', 'bat10', 'bow1', 'bow2', 'bow3', 'bow4', 'bow5', 'bow6', 'runs_off_bat', 'extras', 'Total_score', 'player_dismissed']]
    # Encode the Labels from the existing encoded values
    json_path = Path.joinpath(dataset_path, 'label_encode.json')
    with open(json_path) as f:
        data = json.load(f)
    condition = False

    for col in df_model.columns:
        if col in data.keys():
            condition = True
            col = col
        elif col in ['bat1', 'bat2',
                     'bat3', 'bat4', 'bat5', 'bat6', 'bat7', 'bat8', 'bat9', 'bat10']:
            condition = True
            col = 'Players' #'batsmen'
        elif col in ['bow1',
                     'bow2', 'bow3', 'bow4', 'bow5', 'bow6']:
            col = 'Players' #'bowlers'
            condition = True

        if condition:
            condition = False
            for key in data[col]:
                df_model = df_model.replace([key], data[col][key])

    df_model.to_csv(Path.joinpath(dataset_path, '5_model_df.csv'), index=False)
    return df_model

# Utility function to train the model
def fit(num_epochs, model, loss_fn, opt, train_dl,test_dl):
    train_loss_lst,test_loss_lst = [],[]
    # Repeat for given number of epochs
    for epoch in range(num_epochs):
        # Train with batches of data
        for xb, yb in train_dl:
            # 1. Generate predictions
            pred = model(xb)
            # print(pred)
            #import pdb;pdb.set_trace()
            #print(pred);import pdb;pdb.set_trace()
            # 2. Calculate loss
            train_loss = loss_fn(pred, yb)
            train_loss_lst.append(train_loss.item())
            # 3. Compute gradients
            train_loss.backward()
            # 4. Update parameters using gradients
            opt.step()
            # 5. Reset the gradients to zero
            opt.zero_grad()
        
        with torch.no_grad():
            for xb,yb in test_dl:
                pred = model(xb)
                test_loss = loss_fn(pred,yb)
                test_loss_lst.append(test_loss.item())

        # Print the progress
        if (epoch+1) % 10 == 0:
            print('Epoch [{}/{}], Train-Loss: {:.4f} , Test-Loss : {:.4f}'.format(epoch +
                  1, num_epochs, train_loss.item(),test_loss.item()))

    print('#### Train loss is : {}'.format(train_loss_lst))
    print('#### Test loss is : {}'.format(test_loss_lst))
    loss_dict = {'train' : train_loss_lst,'test' : test_loss_lst}
    with open('loss.json','w') as f:
        json.dump(loss_dict,f)


def linear_model(df):
    inputs_df = df[['venue', 'innings', 'batting_team', 'bowling_team', 'bat1', 'bat2', 'bat3', 'bat4', 'bat5', 'bat6',
                    'bat7', 'bat8', 'bat9', 'bat10', 'bow1', 'bow2', 'bow3', 'bow4', 'bow5', 'bow6']]  # .to_numpy() #dtype=float)

    targets_df = df[['Total_score']]  # .to_numpy()#dtype=float)

    print(inputs_df.shape, targets_df.shape)
    mms_input_obj = MinMaxScaler()
    df_input_norm = pd.DataFrame(mms_input_obj.fit_transform(
        inputs_df), columns=inputs_df.columns)

    mms_output_obj = MinMaxScaler()
    df_output_norm = pd.DataFrame(mms_output_obj.fit_transform(
        targets_df), columns=targets_df.columns)

    inputs_np = df_input_norm.to_numpy()
    targets_np = df_output_norm.to_numpy()

    inputs = torch.from_numpy(inputs_np).float()
    targets = torch.from_numpy(targets_np).float()
    print('### Model input shape is : {}\n### Target shape is : {}'.format(
        inputs.shape, targets.shape))
    targets = targets.view(inputs.shape[0], 1)

    # Define dataset
    train_ds = TensorDataset(inputs, targets)
    print('### Length of Dataset is : {}'.format(len(train_ds)))
    train_ds,test_ds = torch.utils.data.random_split(train_ds,[(round(int(len(train_ds)*70)/100)),len(train_ds)-round(int((len(train_ds)*70)/100))])
    import pdb;pdb.set_trace() 


    # Define data loader
    batch_size = 5
    train_dl = DataLoader(train_ds, batch_size, shuffle=True)
    test_dl = DataLoader(test_ds, 2*batch_size, shuffle=True)

    # Define model
    model = nn.Linear(20, 1)
    print(model.weight.shape)
    print(model.bias.shape)

    # Define loss function
    loss_fn = nn.MSELoss()
    # Define optimizer
    opt = torch.optim.SGD(model.parameters(), lr=1e-1)

    fit(100, model, loss_fn, opt, train_dl,test_dl)

    preds = model(inputs)
    mms_output_obj.inverse_transform(targets)

    # pdb.set_trace()
    pred_df = pd.DataFrame(mms_output_obj.inverse_transform(
        preds.detach().numpy()).astype(int), columns=['Predicted Score in 6-Overs'])

    pd.concat([inputs_df, targets_df, pred_df], axis=1).to_csv(
        'model_pred.csv', index=False)


if __name__ == '__main__':
    starttime = time.time()
    downloadDataset()
    df_parsed = parseDataset()
    encodeLabels_save_to_file(df_parsed)

    df_parsed = format_data(df_parsed)
    df_parsed = create_batsmen_bowler_df(df_parsed)
    df_parsed = update_batsmen_bowler_column_names(df_parsed)
    df = model_df(df_parsed)
    linear_model(df)
    endtime = time.time()
    print(endtime - starttime)
