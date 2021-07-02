# 	IPL Powerplay Score Prediction in AWS Lambda
![IPL banner](https://cricketaddictor.gumlet.io/wp-content/uploads/2021/02/153757448_173638941016448_6980867142752435675_n.jpg?compress=true&quality=80&w=1920&dpr=2.6)


We start with a basic understanding of powerplay in IPL, and quickly go through "Powerplay" score prediction --> based on this a predictive score in 20-overs.

## Introduction: 
It's COVID lockdown time. Everyone is worried about the increasing COVID cases, Work From Home (WFH) environment. No way to go out, No parties, No outing, No vacation plans, No gatherings ...... 

Government announces Lockdown : 1.0, 2.0, 3.0, 4.0 .....

At that time every cricket fan had a question. **"Do new Govt. rules in lockdown 4.0 pave the way for IPL 2020 ????"** Finally IPL 2020 happened. Even pandemic situation can't derail the IPL juggernaut.

IPL 2020 earned revenue of : 4000 crores with 
*  35% reduced cost and 
*  25% increase in viewership. 

As a cricket fan I watch all the matches, during that time I observed that **"Powerplay plays a major role"** which is very Important in Teams score prediction.

## Powerplay in IPL:
"Powerplay" in IPL has fielding restrictions in 1<sup>st</sup> 6-overs, ie.. 


* only 2-fielders can stay outside the Inner-circle
  
  ![powerplay_field](https://github.com/sumankanukollu/cricket_scorePredict_in_first6_overs/raw/main/snippets/ppfield.JPG)
* After powerplay, up to 5-fielders can stay outside inner circle & 4-fielders must remain inside the inner circle.

    ![afterPowerplayField](https://github.com/sumankanukollu/cricket_scorePredict_in_first6_overs/raw/main/snippets/no_ppfield.JPG)

## Effect &  Importance:
Powerplay makes the batting comparatively easy. Also it's a trap for the batsmen, as this will get them to take into a risk and loose their wickets in 1<sup>st</sup> 6-overs.
So, these overs are considered as pillars of any teams  victory.  **75% - of winning chance** depends on the Powerplay score. So, every team's expectation from the top 3-batsmen is   **"START THE INNINGS BIG"**



## Steps of this blog are demonstrated in the below figure:

![flowoftheblog](https://github.com/sumankanukollu/cricket_scorePredict_in_first6_overs/raw/main/snippets/blogflow.JPG)

## 1. Loading the Data Set: 
Actually, this step is the data connection layer and for this very simple prototype, we will keep it simple and easy as loading a data set from [cricsheet](*https://cricsheet.org/downloads/ipl_male_csv.zip)

```python
import pandas as pd
import requests,zipfile

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
    elif dataset_path.exists():
        files = [file for file in dataset_path.iterdir() if file.name ==
                 'all_matches.csv']
        if len(files) == 0:
            getCsvFile()
        else:
            print('### File already extracted in given path')

downloadDataset()
```



## **2. Data Frames:** 

Here we will use pandas to load the dataset into pandas object. Can be done by below code:

```python
csv_file=Path.joinpath(dataset_path, filename)

df = pd.read_csv(csv_file,parse_dates=['start_date'],low_memory=False)
df_parsed = df.copy(deep=True)

df.head()
```
Which contains 200664 : rows , 22 : columns

| match_id | season  | start_date | venue                 | innings | ball | batting_team          | bowling_team                | striker     | non_striker | bowler  | runs_off_bat | extras | wides | noballs | byes | legbyes | penalty | wicket_type | player_dismissed | other_wicket_type | other_player_dismissed |
| -------- | ------- | ---------- | --------------------- | ------- | ---- | --------------------- | --------------------------- | ----------- | ----------- | ------- | ------------ | ------ | ----- | ------- | ---- | ------- | ------- | ----------- | ---------------- | ----------------- | ---------------------- |
| 335982   | 2007/08 | 4/18/2008  | M Chinnaswamy Stadium | 1       | 0.1  | Kolkata Knight Riders | Royal Challengers Bangalore | SC Ganguly  | BB McCullum | P Kumar | 0            | 1      |       |         |      | 1       |         |             |                  |                   |                        |
| 335982   | 2007/08 | 4/18/2008  | M Chinnaswamy Stadium | 1       | 0.2  | Kolkata Knight Riders | Royal Challengers Bangalore | BB McCullum | SC Ganguly  | P Kumar | 0            | 0      |       |         |      |         |         |             |                  |                   |                        |
| 335982   | 2007/08 | 4/18/2008  | M Chinnaswamy Stadium | 1       | 0.3  | Kolkata Knight Riders | Royal Challengers Bangalore | BB McCullum | SC Ganguly  | P Kumar | 0            | 1      | 1     |         |      |         |         |             |                  |                   |                        |
| 335982   | 2007/08 | 4/18/2008  | M Chinnaswamy Stadium | 1       | 0.4  | Kolkata Knight Riders | Royal Challengers Bangalore | BB McCullum | SC Ganguly  | P Kumar | 0            | 0      |       |         |      |         |         |             |                  |                   |                        |



## **3. Exploratory Data Analysis (EDA):**

Prior to Analyze data, **"Data Processing"** is considered most important step, in creation of the Learning Model !  

We can easily get tons of Data in form of various Datasets, but to make that data fit for deriving various insights from it, requires a lot of observation, modification, manipulation and numerous other steps. 
![Data processing snippet](https://miro.medium.com/max/1400/1*vOugEJbcFMoO5qU6TAhpsw.jpeg)

When we freshly download a Dataset for our project, the Data it contains is random(most of the time) i.e. not arranged or not filled in the way we need it to be.
Sometimes, it might have
* NULL Values
* Unnecessary Features
* Datatypes not in a proper format. etc…

So, to treat all these shortcomings, we go through a process which is popularly known as “Data Preprocessing’’.

Comming back to our IPL Dataset, we have to do data processing like below, to train model. 

* Here, headers of this dataset, be self-explanatory. 
* Next Identify the "Null" values in the dataset, hopefully there is no null values present in this dataset.
* Here columns 'season' and 'start_date'  are not needed for our prediction. ```So we can drop the columns 'season' and 'start_date' from the dataset.```
* Delete Non-existing teams : ```  'Kochi Tuskers Kerala' 'Pune Warriors','Rising Pune Supergiants', 'Rising Pune Supergiant','Gujarat Lions'    ```

* Replace the old team names with new team name:
    ```
    'Delhi Daredevils'  --> 'Delhi Capitals'
    'Deccan Chargers'   --> 'Sunrisers Hyderabad'
    'Punjab Kings'      --> 'Kings XI Punjab'
    ```
* Correct the venue column with unique names. In this dataset same stadium is being represented as in multiple ways. So identify those and rename.
    ```
    ['M Chinnaswamy Stadium', 'M.Chinnaswamy Stadium']
    ['Brabourne Stadium', 'Brabourne Stadium, Mumbai']
    ['Punjab Cricket Association Stadium, Mohali', 'Punjab Cricket Association IS Bindra Stadium, Mohali', 'Punjab Cricket Association IS Bindra Stadium']
    ['Wankhede Stadium', 'Wankhede Stadium, Mumbai']
    ['Rajiv Gandhi International Stadium, Uppal', 'Rajiv Gandhi International Stadium']
    ['MA Chidambaram Stadium, Chepauk','MA Chidambaram Stadium',      'MA Chidambaram Stadium, Chepauk, Chennai']
    ```
* Rename the column names:
    ```
    'striker'     --> 'batsmen'
    'non-striker' --> 'batsmen_nonstriker'  (This column is not required)
    'bowler'      --> 'bowlers'
    ```
* Create a columns "Total_score" : which reflects the runs through bat and extra runs through wides,byes,noballs,legbyes...etc.  Hence we can drop columns ```['wides', 'noballs', 'byes', 'legbyes', 'penalty', 'wicket_type','other_wicket_type', 'other_player_dismissed']```
* **Select required Data:**
    Here as we are predicting the "powerplay" score we can drop rest of the data,except 1st 6-overs and drop all innings > 2, As there is no scope for 3rd and 4th innings

#### Code:
  ```python
    df_parsed.drop(columns=['season','start_date'],inplace=True)
    non_exist_teams = ['Kochi Tuskers Kerala',
                    'Pune Warriors',
                    'Rising Pune Supergiants',
                    'Rising Pune Supergiant',
                    'Gujarat Lions']

    mask_bat_team = df_parsed['batting_team'].isin(non_exist_teams)

    mask_bow_team = df_parsed['bowling_team'].isin(non_exist_teams)
    df_parsed = df_parsed[~mask_bat_team]
    df_parsed = df_parsed[~mask_bow_team]

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


  ```

  ```python
    df_parsed['Total_score'] = df_parsed.runs_off_bat + df_parsed.extras
  
    df_parsed.drop(columns=['wides', 'noballs', 'byes', 'legbyes', 'penalty', 'wicket_type',
                    'other_wicket_type', 'other_player_dismissed'], axis=1, inplace=True)
  
    df_parsed.head()
  ```

  ```python
  df_parsed[(df_parsed.ball < 6.0) & (df_parsed.innings < 3)]

  ### Total 35 : venue details present 
  ### Total 8  : Batting teams are there
  ### Total 8  : Bowlling teams are there
  ### Batting teams are : ['Kolkata Knight Riders' 'Royal Challengers Bangalore'
  'Chennai Super Kings' 'Kings XI Punjab' 'Rajasthan Royals'
  'Delhi Capitals' 'Sunrisers Hyderabad' 'Mumbai Indians']
  ### Bowling teams are : ['Royal Challengers Bangalore' 'Kolkata Knight Riders' 'Kings XI Punjab'
  'Chennai Super Kings' 'Delhi Capitals' 'Rajasthan Royals'
  'Sunrisers Hyderabad' 'Mumbai Indians']
  ### Shape of data frame after initial cleanup :(173645, 13)
  ```

  Then, our Data frame looks like below with 173645 : rows and , 13 : columns


| match_id | venue  | innings               | ball | batting_team | bowling_team          | batsmen                     | batsmen_non_striker | bowlers     | runs_off_bat | extras | player_dismissed | Total_score |   |
| -------- | ------ | --------------------- | ---- | ------------ | --------------------- | --------------------------- | ------------------- | ----------- | ------------ | ------ | ---------------- | ----------- | - |
| 335982 | M Chinnaswamy Stadium | 1    | 0.1          | Kolkata Knight Riders | Royal Challengers Bangalore | SC Ganguly          | BB McCullum | P Kumar      | 0      | 1                | NaN         | 1 |
| 335982 | M Chinnaswamy Stadium | 1    | 0.2          | Kolkata Knight Riders | Royal Challengers Bangalore | BB McCullum         | SC Ganguly  | P Kumar      | 0      | 0                | NaN         | 0 |
| 335982 | M Chinnaswamy Stadium | 1    | 0.3          | Kolkata Knight Riders | Royal Challengers Bangalore | BB McCullum         | SC Ganguly  | P Kumar      | 0      | 1                | NaN         | 1 |
| 335982 | M Chinnaswamy Stadium | 1    | 0.4          | Kolkata Knight Riders | Royal Challengers Bangalore | BB McCullum         | SC Ganguly  | P Kumar      | 0      | 0                | NaN         | 0 |
| 335982 | M Chinnaswamy Stadium | 1    | 0.5          | Kolkata Knight Riders | Royal Challengers Bangalore | BB McCullum         | SC Ganguly  | P Kumar      | 0      | 0                | NaN         | 0 |






## **4. Encoding:**
  If the data set loaded has the columns with the labels, then the label encoder solves the transformation from string to numerical values and save the details in a file.

   In our dataset, we have labels for the below columns.
  ```
    1   venue                54525 non-null  object 
    4   batting_team         54525 non-null  object 
    5   bowling_team         54525 non-null  object 
    6   batsmen              54525 non-null  object 
    7   batsmen_non_striker  54525 non-null  object 
    8   bowlers              54525 non-null  object 
  ```

### Code:
```python
  label_encode_dict = {}

  le = LabelEncoder()
  le.fit(players_df.Players)
  Players_e = le.transform(players_df.Players)
  Players_e_inv = le.inverse_transform(Players_e)

  label_encode_dict['Players'] = dict(
        zip(Players_e_inv, map(int, Players_e)))

  le.fit(df_parsed.venue)
  venue_e = le.transform(df_parsed.venue)
  venue_e_inv = le.inverse_transform(venue_e)
  label_encode_dict['venue'] = dict(zip(venue_e_inv, map(int, venue_e)))

  le.fit(df_parsed.batting_team)
  batting_team_e = le.transform(df_parsed.batting_team)
  batting_team_e_inv = le.inverse_transform(batting_team_e)
  label_encode_dict['batting_team'] = dict(
    zip(batting_team_e_inv, map(int, batting_team_e)))

  le.fit(df_parsed.bowling_team)
  bowling_team_e = le.transform(df_parsed.bowling_team)
  bowling_team_e_inv = le.inverse_transform(bowling_team_e)
  label_encode_dict['bowling_team'] = dict(
    zip(bowling_team_e_inv, map(int, bowling_team_e)))

  with open(json_path, 'w') as f:
    json.dump(label_encode_dict, f)
```
* **Format dataset :**
  Next grab all the players who played in 6-overs (bat and bowl) and arrange across in multiple columns say [bat1,bat2,bat3,.....bat10,bow1,bo2,bo3,....bow5,bow6] in the same sequence from the dataset (for better prediction).

  During powerplay 10-batsmen can play if they loose wickets as well as only 6-bowlers can bowl 1-over each. 
  So, I created columns for 10-batsmen and 6-bowlers

  Now the data frame looks like :
  
  | venue | innings | batting_team | bowling_team | bat1 | bat2 | bat3 | bat4 | bat5 | bat6 | bat7 | bat8 | bat9 | bat10 | bow1 | bow2 | bow3 | bow4 | bow5 | bow6 | runs_off_bat | extras | Total_score | player_dismissed |
  | ----- | ------- | ------------ | ------------ | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ----- | ---- | ---- | ---- | ---- | ---- | ---- | ------------ | ------ | ----------- | ---------------- |
  | 15    | 1       | 3            | 6            | 419  | 70   | 385  | 0.0  | 0.0  | 0.0  | 0.0  | 0.0  | 0.0  | 0.0   | 326  | 512  | 19   | 0.0  | 0.0  | 0.0  | 51           | 10     | 61          | 1                |
  | 15    | 2       | 6            | 3            | 351  | 496  | 483  | 187  | 97   | 297  | 0.0  | 0.0  | 0.0  | 0.0   | 22   | 163  | 20   | 0.0  | 0.0  | 0.0  | 19           | 7      | 26          | 4                |
  | 23    | 1       | 0            | 2            | 331  | 286  | 274  | 0.0  | 0.0  | 0.0  | 0.0  | 0.0  | 0.0  | 0.0   | 65   | 408  | 201  | 0.0  | 0.0  | 0.0  | 50           | 3      | 53          | 1                |
  | 23    | 2       | 2            | 0            | 207  | 201  | 0.0  | 0.0  | 0.0  | 0.0  | 0.0  | 0.0  | 0.0  | 0.0   | 184  | 296  | 0.0  | 0.0  | 0.0  | 0.0  | 61           | 2      | 63          | 1                |
  | 10    | 1       | 5            | 1            | 468  | 506  | 443  | 255  | 0.0  | 0.0  | 0.0  | 0.0  | 0.0  | 0.0   | 146  | 62   | 275  | 0.0  | 0.0  | 0.0  | 38           | 2      | 40          | 2                |



## **5. Normalization:**

In Encoding step all data has been converted into numerical values, which are in different ranges. In this Normalization step, we represent the numerical column values in a common data scale, without loosing information & distorting differences in the ranges of values, with the help of "MinMaxScaler".

```python
inputs_df  = df[['venue', 'innings', 'batting_team', 'bowling_team', 'bat1', 'bat2', 'bat3', 'bat4', 'bat5', 'bat6','bat7', 'bat8', 'bat9', 'bat10', 'bow1', 'bow2', 'bow3', 'bow4', 'bow5', 'bow6']]  # .to_numpy() #dtype=float)

targets_df = df[['Total_score']]  # .to_numpy()#dtype=float)


mms_input_obj = MinMaxScaler()
df_input_norm = pd.DataFrame(mms_input_obj.fit_transform(inputs_df), columns=inputs_df.columns)

mms_output_obj = MinMaxScaler()
df_output_norm = pd.DataFrame(mms_output_obj.fit_transform(targets_df), columns=targets_df.columns)

inputs_np  = df_input_norm.to_numpy()
targets_np = df_output_norm.to_numpy()

inputs = torch.from_numpy(inputs_np).float()
targets = torch.from_numpy(targets_np).float()

targets = targets.view(inputs.shape[0], 1)
```
| venue   | innings | batting_team | bowling_team | bat1     | bat2     | bat3     | bat4     | bat5     | bat6     | bat7 | bat8 | bat9 | bat10 | bow1     | bow2     | bow3     | bow4 | bow5 | bow6 |
| ------- | ------- | ------------ | ------------ | -------- | -------- | -------- | -------- | -------- | -------- | ---- | ---- | ---- | ----- | -------- | -------- | -------- | ---- | ---- | ---- |
| 0.43750 | 0.0     | 0.285714     | 0.714286     | 0.803150 | 0.140594 | 0.739726 | 0.000000 | 0.000000 | 0.000000 | 0.0  | 0.0  | 0.0  | 0.0   | 0.621569 | 1.000000 | 0.035225 | 0.0  | 0.0  | 0.0  |
| 0.43750 | 1.0     | 0.714286     | 0.285714     | 0.673228 | 0.972277 | 0.941292 | 0.372549 | 0.194118 | 0.574257 | 0.0  | 0.0  | 0.0  | 0.0   | 0.039216 | 0.324803 | 0.037182 | 0.0  | 0.0  | 0.0  |
| 0.65625 | 0.0     | 0.000000     | 0.428571     | 0.633858 | 0.546535 | 0.524462 | 0.000000 | 0.000000 | 0.000000 | 0.0  | 0.0  | 0.0  | 0.0   | 0.129412 | 0.779528 | 0.395303 | 0.0  | 0.0  | 0.0  |
| 0.65625 | 1.0     | 0.428571     | 0.000000     | 0.401575 | 0.396040 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.0  | 0.0  | 0.0  | 0.0   | 0.364706 | 0.561024 | 0.000000 | 0.0  | 0.0  | 0.0  |
| 0.28125 | 0.0     | 0.857143     | 0.142857     | 0.907480 | 0.996040 | 0.851272 | 0.486275 | 0.000000 | 0.000000 | 0.0  | 0.0  | 0.0  | 0.0   | 0.294118 | 0.100394 | 0.526419 | 0.0  | 0.0  | 0.0  |

## **6. Split Test & Train:**









