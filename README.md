# IPL Powerplay Score Prediction using AWS Lambda

![IPL banner](https://cricketaddictor.gumlet.io/wp-content/uploads/2021/02/153757448_173638941016448_6980867142752435675_n.jpg?compress=true&quality=80&w=1920&dpr=2.6)



We start with a basic understanding of powerplay in IPL, and quickly go through "Powerplay" score prediction walk through & Hyper Parameters Optimization --> based on this a predictive score in 20-overs.

You want to access code? [Click here](https://github.com/sumankanukollu/cricket_scorePredict_in_first6_overs/blob/main/IPL_Powerplay_Score_Prediction_v2.ipynb)

You want to Predict Score? [Click Here](https://suman-projects.s3.ap-south-1.amazonaws.com/myWebPage/index.html)

**Disclaimer:** The analysis and prediction done here is for learning purpose only and should not be used for any illegal activities such as betting.

## Project Demo
![demo](https://github.com/sumankanukollu/cricket_scorePredict_in_first6_overs/raw/main/Inference_Recoding.gif)


## Introduction: 
It's COVID lockdown time. Everyone is worried about the increasing COVID cases, Work From Home (WFH) environment. No way to go out, No parties, No outing, No vacation plans, No gatherings ...... 

Government announces Lockdown : 1.0, 2.0, 3.0, 4.0 .....

At that time every cricket fan had a question. **"Do new Govt. rules in lockdown 4.0 pave the way for IPL 2020 ??"** Finally IPL 2020 happened. Even pandemic situation can't derail the IPL juggernaut.

IPL 2020 earned revenue of 4000 crores with, 
*  35% reduced cost and 
*  25% increase in viewership. 

As a cricket fan I watch all the matches, during that time I observed that **"Powerplay plays a major role"** which is very Important in Teams score prediction.

## Powerplay in IPL:
"Powerplay" in IPL has fielding restrictions in 1<sup>st</sup> 6-overs, ie.. 


* only 2-fielders can stay outside the Inner-circle
  
  <img src="https://github.com/sumankanukollu/cricket_scorePredict_in_first6_overs/raw/main/snippets/ppfield.JPG" alt="powerplay_field" style="zoom:50%;" />
* After powerplay, up to 5-fielders can stay outside inner circle & 4-fielders must remain inside the inner circle.

    <img src="https://github.com/sumankanukollu/cricket_scorePredict_in_first6_overs/raw/main/snippets/no_ppfield.JPG" alt="afterPowerplayField" style="zoom:50%;" />

## Effect &  Importance:
Powerplay makes the batting comparatively easy. Also it's a trap for the batsmen, as this will get them to take into a risk and loose their wickets in 1<sup>st</sup> 6-overs.
So, these overs are considered as pillars of any teams  victory.  **75% - of winning chance** depends on the Powerplay score. So, every team's expectation from the top 3-batsmen is   **"START THE INNINGS BIG"**



## Steps of this blog are demonstrated in the below figure:

<img src="https://github.com/sumankanukollu/cricket_scorePredict_in_first6_overs/raw/main/snippets/blogflow.JPG" alt="flowoftheblog" style="zoom: 50%;" />

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

## 2. Data Frames:

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

## 3. Exploratory Data Analysis (EDA):

Prior to Analyze data, **"Data Processing"** is considered most important step, in creation of the Learning Model !  

We can easily get tons of Data in form of various Datasets, but to make that data fit for deriving various insights from it, requires a lot of observation, modification, manipulation and numerous other steps. 
<img src="https://miro.medium.com/max/1400/1*vOugEJbcFMoO5qU6TAhpsw.jpeg" alt="Data processing snippet" style="zoom:70%;" />

When we freshly download a Dataset for our project, the Data it contains is random(most of the time) i.e. not arranged or not filled in the way we need it to be.
Sometimes, it might have `NULL Values, Unnecessary Features, Datatypes not in a proper format. etc…`

So, to treat all these shortcomings, we go through a process which is popularly known as “Data Preprocessing’’.

Coming back to our IPL Dataset, we have to do data processing like below, to train model. 

Here, headers of this dataset,are self-explanatory. 

### Data Processing Steps:

* First `Identify the "Null" values` in the dataset, hopefully there is no null values present in this dataset for the **required fields**.
```python
df.isnull().sum()
    match_id                       0
    season                         0
    start_date                     0
    venue                          0
    innings                        0
    ball                           0
    batting_team                   0
    bowling_team                   0
    striker                        0
    non_striker                    0
    bowler                         0
    runs_off_bat                   0
    extras                         0
    wides                     194583
    noballs                   199854
    byes                      200135
    legbyes                   197460
    penalty                   200662
    wicket_type               190799
    player_dismissed          190799
    other_wicket_type         200664
    other_player_dismissed    200664
    dtype: int64
```

* **Select the 1<sup>st</sup> 6-Overs match details & drop rest:**
  
    * Here as we are predicting the "powerplay" score we can drop rest of the data, except 1st 6-overs and 
    * drop all innings > 2, 
    As there is no scope for 3rd and 4th innings
    ```python
         df_parsed[(df_parsed.ball < 6.0) & (df_parsed.innings < 3)]
    ```

* **Identify & Drop Innings played < 6 overs :**

    * In our data few matches are not conducted and declared as "no result" may be due to rain... some technical problems.
    * Either match result is declared by DCB method, we can't predict the score.

    Which may lead to outliers, to our model. Because If team played only 3 overs, then score minimum value effects which leads to modify the following "Normalization" process. So better to drop these data in our prediction.
```python
obj = df_parsed.query('ball<6.0&innings<3').groupby(
    ['match_id', 'venue', 'innings', 'batting_team', 'bowling_team'])

for key, val in obj:
    if val['ball'].count() < 36:
        df_parsed.drop(labels=val.index, axis=0, inplace=True)
```

* Here columns 'season' and 'start_date'  are not needed for our prediction. `So we can drop the columns 'season' and 'start_date' from the dataset.`
```python
df_parsed.drop(columns=['season', 'start_date'], inplace=True)
```

* Delete Non-existing teams : 
```python
non_exist_teams = ['Kochi Tuskers Kerala',
                   'Pune Warriors',
                   'Rising Pune Supergiants',
                   'Rising Pune Supergiant',
                   'Gujarat Lions']

mask_bat_team = df_parsed['batting_team'].isin(non_exist_teams)

mask_bow_team = df_parsed['bowling_team'].isin(non_exist_teams)
df_parsed = df_parsed[~mask_bat_team]
df_parsed = df_parsed[~mask_bow_team]
```

* Replace the old team names with new team name:
```python
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
```

* Correct the venue column with unique names. In this dataset same stadium is being represented as in multiple ways. So identify those and rename.
```python
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

* Create a columns `"Total_score" : which reflects the runs through bat and extra runs` through wide's, byes,no-balls,leg byes... etc. 
```python
df_parsed['Total_score'] = df_parsed.runs_off_bat + df_parsed.extras

df_parsed.drop(columns=['wides', 'noballs', 'byes', 'legbyes', 'penalty', 'wicket_type',
                        'other_wicket_type', 'other_player_dismissed'], axis=1, inplace=True)
```
 * Can Ignore  considering columns in a data frame while propcessing  `['wides', 'noballs', 'byes', 'legbyes', 'penalty', 'wicket_type','other_wicket_type', 'other_player_dismissed']`

#### Data Frame looks like :
Then, our Data frame looks like below with 54525 : rows, 13 : columns (Initially it was 200664 : rows , 22 : columns)

```python
# Total 33 : venue details present
# Total 8  : Batting teams are there
# Total 8  : Bowling teams are there
# Batting teams are : ['Kolkata Knight Riders' 'Royal Challengers Bangalore'
 'Chennai Super Kings' 'Kings XI Punjab' 'Rajasthan Royals'
 'Delhi Capitals' 'Sunrisers Hyderabad' 'Mumbai Indians']
# Bowling teams are : ['Royal Challengers Bangalore' 'Kolkata Knight Riders' 'Kings XI Punjab'
 'Chennai Super Kings' 'Delhi Capitals' 'Rajasthan Royals'
 'Sunrisers Hyderabad' 'Mumbai Indians']
# Shape of data frame after initial cleanup :(54525, 13)
```

| match_id | venue                 | innings | ball | batting_team          | bowling_team                | batsmen     | batsmen_non_striker | bowlers | runs_off_bat | extras | player_dismissed | Total_score |
| -------- | --------------------- | ------- | ---- | --------------------- | --------------------------- | ----------- | ------------------- | ------- | ------------ | ------ | ---------------- | ----------- |
| 335982   | M Chinnaswamy Stadium | 1       | 0.1  | Kolkata Knight Riders | Royal Challengers Bangalore | SC Ganguly  | BB McCullum         | P Kumar | 0            | 1      | NaN              | 1           |
| 335982   | M Chinnaswamy Stadium | 1       | 0.2  | Kolkata Knight Riders | Royal Challengers Bangalore | BB McCullum | SC Ganguly          | P Kumar | 0            | 0      | NaN              | 0           |
| 335982   | M Chinnaswamy Stadium | 1       | 0.3  | Kolkata Knight Riders | Royal Challengers Bangalore | BB McCullum | SC Ganguly          | P Kumar | 0            | 1      | NaN              | 1           |
| 335982   | M Chinnaswamy Stadium | 1       | 0.4  | Kolkata Knight Riders | Royal Challengers Bangalore | BB McCullum | SC Ganguly          | P Kumar | 0            | 0      | NaN              | 0           |
| 335982   | M Chinnaswamy Stadium | 1       | 0.5  | Kolkata Knight Riders | Royal Challengers Bangalore | BB McCullum | SC Ganguly          | P Kumar | 0            | 0      | NaN              | 0           |

## **4. Encoding:**
Till now as a initial step,

* We cleaned our dataset with all the "null" values and 

* filtered the columns/rows data (which are not used for prediction) and

* Added the required column with values (like total_score) to make the dataset clean.

As you can see above, our cleaned dataset is having 13 columns of multiple Dtype like int64 and float64 and object.

So next what I am going to do is trying to convert all these multiple Dtypes into a single Dtype, to train the model.

  ```python
    1   venue                54525 non-null  object 
    4   batting_team         54525 non-null  object 
    5   bowling_team         54525 non-null  object 
    6   batsmen              54525 non-null  object 
    7   bowlers              54525 non-null  object 
  ```

### 4.1. Encode "batsmen" and "bowlers" column values:
  * Here we can see, few players who can bat as well as bowl.
    Means same player will be listed as a batsmen and as well as bowler.
  * So to make the prediction properly, I am creating one Data frame with all the players name it as "players_df", which I use for encoding the players with some value to identify.
  * For inference going further, I create a dictionary with all these encoded values 

```python
players_df = pd.DataFrame(np.append(
                          df_parsed.batsmen.unique(), df_parsed.bowlers.unique()),
                          columns=['Players']
                          )

label_encode_dict = {}

dct = dict(enumerate(players_df.Players.astype('category').cat.categories))
label_encode_dict['Players'] = dict(zip(dct.values(), dct.keys()))
```
### 4.2. Encode "venue" and "batting_team" and "bowling_team" column values:
```python
for col in ['venue', 'batting_team', 'bowling_team']:
    dct = dict(enumerate(df_parsed[col].astype('category').cat.categories))
    label_encode_dict[col] = dict(zip(dct.values(), dct.keys()))
```
### 4.3. Save the encoded values to a json file:
```python
label_encode_dict['Total_score_min'] = float(Total_score_min_max[0])
label_encode_dict['Total_score_max'] = float(Total_score_min_max[1])
label_encode_dict['Total_score_min'],label_encode_dict['Total_score_max']

with open(json_path, 'w') as f:
    json.dump(label_encode_dict, f)
```

### 4.4. Format the Dataset:
In this step I am trying to club the all rows with respect to matchID and Innings (as match ID is unique way to identify a particular match and Innings to identify who bat first).

Based on these two details, 
* grab all the batsmen and bowler details who batted and bowled in 1<sup>st</sup> 6-overs
* Calculate the total score (runs through bat + extra runs)
* How many players dismissed in 1st 6-overs
```python 
print('### Shape of Dataframe before format_data : {}'.format(df_parsed.shape))

Runs_off_Bat_6_overs = df_parsed.groupby(['match_id', 'venue', 'innings', 'batting_team', 'bowling_team'])['runs_off_bat'].sum()

Extras_6_overs = df_parsed.groupby(['match_id', 'venue', 'innings', 'batting_team', 'bowling_team'])['extras'].sum()

TotalScore_6_overs = df_parsed.groupby(['match_id', 'venue', 'innings', 'batting_team', 'bowling_team'])['Total_score'].sum()

Total_WktsDown = df_parsed.groupby(['match_id', 'venue', 'innings', 'batting_team', 'bowling_team'])['player_dismissed'].count()

bat_df = df_parsed.groupby(['match_id', 'venue', 'innings','batting_team', 'bowling_team'])['batsmen'].apply(list)

bow_df = df_parsed.groupby(['match_id', 'venue', 'innings','batting_team', 'bowling_team'])['bowlers'].apply(list)

df_parsed = pd.DataFrame(pd.concat([bat_df, bow_df, Runs_off_Bat_6_overs, Extras_6_overs, TotalScore_6_overs, Total_WktsDown],axis=1)).reset_index()
```

### 4.5. Align the batsmen and bowlers details in to a separate column

In above formatted dataset, we got list of batsmen and bowlers details who batted and bowled in 6-overs.

Now we have to arrange these batsmen into a separate columns,
```
 * say bat1,bat2,bat3,bat4....bat10
 * say bow1,bow2,bow3.....bow6
```

Here I selected only 10-batsmen (as we have only 10-wickets), and 6-bowlers (can bowl in 6-overs) because in 6-overs this is only possible.

```
For proper prediction the order of batsmen and bowlers given the dataset matters. So we need to keep the order :
* batsmen : who batted 1st,2nd 3rd and 4th ... wicket same
* bowler  : who bowled in 1st,2nd,3rd,4th,5th and 6th overs same
```

### 4.6. Create a batsmen and bowlers dummy data frame:
To keep track of the order same, so 1st I am going to create a dummy data frame 
* with 10-batsmen with column names [bat1,bat2,.....bat9,bat10]
* with 6-bowlers with column names [bow1,bow2,bow3,bow4,bow5,bow6]
```python
bat  = pd.DataFrame(np.zeros((df_parsed.shape[0],10),dtype=float),columns=['bat1', 'bat2', 'bat3', 'bat4', 'bat5', 'bat6', 'bat7', 'bat8', 'bat9', 'bat10'])
bowl = pd.DataFrame(np.zeros((df_parsed.shape[0],6),dtype=float),columns=['bow1', 'bow2', 'bow3', 'bow4', 'bow5', 'bow6'])

columns = ['bat1', 'bat2', 'bat3', 'bat4', 'bat5', 'bat6', 'bat7', 'bat8', 'bat9', 'bat10','bow1', 'bow2', 'bow3', 'bow4', 'bow5', 'bow6']
df_bat_bow = pd.concat([bat,bowl],axis=1)
df_bat_bow

```
| bat1 | bat2 | bat3 | bat4 | bat5 | bat6 | bat7 | bat8 | bat9 | bat10 | bow1 | bow2 | bow3 | bow4 | bow5 | bow6 |      |
| ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ----- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| 0    | 0.0  | 0.0  | 0.0  | 0.0  | 0.0  | 0.0  | 0.0  | 0.0  | 0.0   | 0.0  | 0.0  | 0.0  | 0.0  | 0.0  | 0.0  | 0.0  |
| 1    | 0.0  | 0.0  | 0.0  | 0.0  | 0.0  | 0.0  | 0.0  | 0.0  | 0.0   | 0.0  | 0.0  | 0.0  | 0.0  | 0.0  | 0.0  | 0.0  |
| 2    | 0.0  | 0.0  | 0.0  | 0.0  | 0.0  | 0.0  | 0.0  | 0.0  | 0.0   | 0.0  | 0.0  | 0.0  | 0.0  | 0.0  | 0.0  | 0.0  |
| 3    | 0.0  | 0.0  | 0.0  | 0.0  | 0.0  | 0.0  | 0.0  | 0.0  | 0.0   | 0.0  | 0.0  | 0.0  | 0.0  | 0.0  | 0.0  | 0.0  |
| 4    | 0.0  | 0.0  | 0.0  | 0.0  | 0.0  | 0.0  | 0.0  | 0.0  | 0.0   | 0.0  | 0.0  | 0.0  | 0.0  | 0.0  | 0.0  | 0.0  |


| 1453 rows × 16 columns |  

```python
# Concat the this dummy data frame with the parsed data frame
df_parsed =  pd.concat([df_parsed, df_bat_bow], axis=1)
```

### 4.7. Update the batsmen list of elements into each column in the same order:

Means, data which is in list of elements in each matchID and innings into corresponding individual batsmen columns in the same order. 
```python
Example, here in below list 1st batsmen is bat1 -> SC Ganguly, 2nd is bat2 -> BB McCullum, 3rd is bat3 -> RT Poting ...etc like 

"['SC Ganguly', 'BB McCullum', 'BB McCullum', 'BB McCullum', 'BB McCullum', 'BB McCullum', 'BB McCullum', 'BB McCullum', 'BB McCullum', 'BB McCullum', 'BB McCullum', 'BB McCullum', 'BB McCullum', 'SC Ganguly', 'SC Ganguly', 'SC Ganguly', 'BB McCullum', 'BB McCullum', 'SC Ganguly', 'BB McCullum', 'BB McCullum', 'BB McCullum', 'SC Ganguly', 'SC Ganguly', 'SC Ganguly', 'BB McCullum', 'SC Ganguly', 'SC Ganguly', 'BB McCullum', 'BB McCullum', 'BB McCullum', 'SC Ganguly', 'BB McCullum', 'SC Ganguly', 'RT Ponting', 'RT Ponting', 'RT Ponting', 'RT Ponting', 'BB McCullum', 'RT Ponting', 'BB McCullum', 'RT Ponting', 'RT Ponting', 'BB McCullum', 'BB McCullum', 'BB McCullum', 'RT Ponting', 'BB McCullum', 'RT Ponting', 'BB McCullum', 'BB McCullum', 'BB McCullum', 'BB McCullum', 'BB McCullum', 'RT Ponting', 'BB McCullum', 'RT Ponting', 'BB McCullum', 'RT Ponting', 'BB McCullum', 'BB McCullum', 'BB McCullum', 'BB McCullum', 'RT Ponting', 'RT Ponting', 'RT Ponting', 'RT Ponting', 'RT Ponting', 'BB McCullum', 'BB McCullum', 'BB McCullum', 'BB McCullum', 'RT Ponting', 'RT Ponting', 'RT Ponting', 'BB McCullum', 'BB McCullum', 'BB McCullum', 'BB McCullum', 'BB McCullum', 'BB McCullum', 'BB McCullum', 'DJ Hussey', 'DJ Hussey', 'BB McCullum', 'DJ Hussey', 'BB McCullum', 'DJ Hussey', 'DJ Hussey', 'BB McCullum', 'BB McCullum', 'BB McCullum', 'DJ Hussey', 'BB McCullum', 'DJ Hussey', 'DJ Hussey', 'BB McCullum', 'BB McCullum', 'BB McCullum', 'DJ Hussey', 'BB McCullum', 'DJ Hussey', 'DJ Hussey', 'BB McCullum', 'BB McCullum', 'BB McCullum', 'DJ Hussey', 'BB McCullum', 'Mohammad Hafeez', 'Mohammad Hafeez', 'BB McCullum', 'BB McCullum', 'BB McCullum', 'BB McCullum', 'BB McCullum', 'BB McCullum', 'BB McCullum', 'Mohammad Hafeez', 'BB McCullum', 'BB McCullum', 'BB McCullum', 'BB McCullum', 'BB McCullum', 'BB McCullum']"

df_parsed['batsmen'] = df_parsed.batsmen.map(OrderedDict.fromkeys).apply(list)
df_parsed['bowlers'] = df_parsed.bowlers.map(OrderedDict.fromkeys).apply(list)

for row,val in enumerate(df_parsed.batsmen):
    for i in range(len(val)):
        df_parsed.loc[row,'bat%s'%(i+1)] = val[i]

for row,val in enumerate(df_parsed.bowlers):
    for i in range(len(val)):
        df_parsed.loc[row,'bow%s'%(i+1)] = val[i]

df_parsed.loc[:,['bat1','bat2','bat3','bat4','bat5','bat6','bat7','bat8','bat9','bat10','bow1','bow2','bow3','bow4','bow5','bow6']]
```

So finally our dataframe is ready with batsmen and bowlers details.
So we can drop few columns which are not important.

```python
df_model = df_parsed[['venue', 'innings', 'batting_team', 'bowling_team', 'bat1', 'bat2', 'bat3', 'bat4', 'bat5', 'bat6', 'bat7', 'bat8','bat9', 'bat10', 'bow1', 'bow2', 'bow3', 'bow4', 'bow5', 'bow6', 'runs_off_bat', 'extras', 'Total_score', 'player_dismissed']]

df_model.columns

Index(['venue', 'innings', 'batting_team', 'bowling_team', 'bat1', 'bat2',
       'bat3', 'bat4', 'bat5', 'bat6', 'bat7', 'bat8', 'bat9', 'bat10', 'bow1',
       'bow2', 'bow3', 'bow4', 'bow5', 'bow6', 'runs_off_bat', 'extras',
       'Total_score', 'player_dismissed'],
      dtype='object')
```

### 4.8. Encode the multiple Dtypes into single Dtype:

Now its time to use, the label encoded values (already done in previous steps) to encode the data frame.

```python
json_path = Path.joinpath(dataset_path, 'label_encode.json')
with open(json_path) as f:
    data = json.load(f)

condition = False

for col in df_model.columns:
    if col in data.keys():
        condition = True
        col = col
    elif col in ['bat1', 'bat2', 'bat3', 'bat4', 'bat5', 'bat6', 'bat7', 'bat8', 'bat9', 'bat10']:
        condition = True
        col = 'Players'  # 'batsmen'
    elif col in ['bow1', 'bow2', 'bow3', 'bow4', 'bow5', 'bow6']:
        col = 'Players'  # 'bowlers'
        condition = True

    if condition:
        condition = False
        for key in data[col]:
            df_model = df_model.replace([key], data[col][key])
```

### 4.9. At the end of this step our data frame looks like :

| venue | innings | batting_team | bowling_team | bat1 | bat2 | bat3 | bat4 | bat5 | bat6 | bat7 | bat8 | bat9 | bat10 | bow1 | bow2 | bow3 | bow4 | bow5 | bow6 | runs_off_bat | extras | Total_score | player_dismissed |
| ----- | ------- | ------------ | ------------ | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ----- | ---- | ---- | ---- | ---- | ---- | ---- | ------------ | ------ | ----------- | ---------------- |
| 14    | 1       | 3            | 6            | 419  | 70   | 385  | 0    | 0    | 0    | 0    | 0    | 0    | 0     | 326  | 512  | 19   | 0    | 0    | 0    | 51           | 10     | 61          | 1                |
| 14    | 2       | 6            | 3            | 351  | 496  | 483  | 187  | 97   | 297  | 0    | 0    | 0    | 0     | 22   | 163  | 20   | 0    | 0    | 0    | 19           | 7      | 26          | 4                |
| 21    | 1       | 0            | 2            | 331  | 286  | 274  | 0    | 0    | 0    | 0    | 0    | 0    | 0     | 65   | 408  | 201  | 0    | 0    | 0    | 50           | 3      | 53          | 1                |

## 5. Normalization:
In Encoding step all data has been converted into numerical values, which are in different ranges. 

In this Normalization step, we represent the numerical column values in a common data scale, without loosing information & distorting differences in the ranges of values, with the help of **"MinMaxScaler"**.

```python
df_model = df_model.applymap(np.float64)
df_norm = (df_model - df_model.min())/(df_model.max() - df_model.min())

df_norm.fillna(0.0,inplace=True)

df_norm.head()
```
At the end of this step data frame looks like :

| venue   | innings | batting_team | bowling_team | bat1     | bat2     | bat3     | bat4     | bat5     | bat6     | ...  | bow1     | bow2     | bow3     | bow4 | bow5 | bow6 | runs_off_bat | extras   | Total_score | player_dismissed |
| ------- | ------- | ------------ | ------------ | -------- | -------- | -------- | -------- | -------- | -------- | ---- | -------- | -------- | -------- | ---- | ---- | ---- | ------------ | -------- | ----------- | ---------------- |
| 0.43750 | 0.0     | 0.428571     | 0.857143     | 0.819253 | 0.134387 | 0.751953 | 0.000000 | 0.000000 | 0.000000 | ...  | 0.636008 | 1.000000 | 0.037109 | 0.0  | 0.0  | 0.0  | 0.452632     | 0.666667 | 0.516484    | 0.2              |
| 0.43750 | 1.0     | 0.857143     | 0.428571     | 0.685658 | 0.976285 | 0.943359 | 0.365949 | 0.189824 | 0.586957 | ...  | 0.041096 | 0.314342 | 0.039062 | 0.0  | 0.0  | 0.0  | 0.115789     | 0.466667 | 0.131868    | 0.8              |
| 0.65625 | 0.0     | 0.000000     | 0.285714     | 0.646365 | 0.561265 | 0.535156 | 0.000000 | 0.000000 | 0.000000 | ...  | 0.125245 | 0.795678 | 0.392578 | 0.0  | 0.0  | 0.0  | 0.442105     | 0.200000 | 0.428571    | 0.2              |
| 0.65625 | 1.0     | 0.285714     | 0.000000     | 0.402750 | 0.393281 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | ...  | 0.358121 | 0.575639 | 0.000000 | 0.0  | 0.0  | 0.0  | 0.557895     | 0.133333 | 0.538462    | 0.2              |
| 0.28125 | 0.0     | 0.714286     | 0.142857     | 0.915521 | 0.996047 | 0.865234 | 0.499022 | 0.000000 | 0.000000 | ...  | 0.283757 | 0.115914 | 0.537109 | 0.0  | 0.0  | 0.0  | 0.315789     | 0.133333 | 0.285714    | 0.4              |


## 6. Split Train & test data:

Split the dataset into 70% for train and 30% for test
![splitDatasetImg](https://miro.medium.com/max/1400/1*n7Ob33nRMq07BZPbNtfItw.png)

  1. Identify the inputs vs targets from the pandas dataframe  and convert into Torch Tensor
  2. Create the torch dataset
  3. Now split the torch_ds into train_ds and test_ds datasets (70% vs 30%)
  4. Create a dataloader for train_ds and test_ds

```python
inputs = torch.Tensor(df_norm.iloc[:,:-4].values.astype(np.float32))
targets = torch.Tensor(df_norm.loc[:,'Total_score'].values.reshape(-1,1).astype(np.float32))

torch_ds = torch.utils.data.TensorDataset(inputs,targets)
len(torch_ds) # 1453

train_ds_sz = int(round((len(torch_ds)*split_ratio[0])/100.0))
test_ds_sz  = len(torch_ds) - train_ds_sz

train_ds,test_ds = torch.utils.data.random_split(torch_ds,[train_ds_sz,test_ds_sz])
len(train_ds),len(test_ds) # (1017, 436)

train_dl = torch.utils.data.DataLoader(dataset = train_ds,batch_size=train_bs, shuffle = True)
test_dl = torch.utils.data.DataLoader(dataset = test_ds,batch_size=2*train_bs)

```

## 7. Model Selection:

Here I am using Linear regression model.
```python
class linearRegression(torch.nn.Module):
    def __init__(self, inputSize, outputSize):
        super(linearRegression, self).__init__()
        self.linear = torch.nn.Linear(inputSize, outputSize)

    def forward(self, x):
        out = self.linear(x)
        return out

model = linearRegression(inputs.shape[1],targets.shape[1]) # (20,1)
```

## 8. Hyper Parameters Optimization:
I tried many combinations with respect to LR,momentum,NAG, batch size and optimizers and schedulers to find the global minima.

All the results are available at [record metrics](https://github.com/sumankanukollu/cricket_scorePredict_in_first6_overs/blob/main/Recorded_Metrics.csv)

Out of all tests, Identified below Hyper Parameters gives better results:
```
1. Train vs Test Batch Sz : 32 & 64
2. Loss : MSE Loss()
3. Optimizer : "SGD (Parameter Group 0,
                        dampening: 0,
                        lr: 0.1,
                        momentum: 0.9,
                        nesterov: True,
                        weight_decay: 0
                    )"
4. Minimum Train Loss "0.0061" observed in 36th epoch (can see in below graph)
```

```python
#loss = torch.nn.L1Loss()
loss = torch.nn.MSELoss()
loss_Description = str(loss) # Its declared in Hyper Params section in top

#opt  = torch.optim.SGD(model.parameters(),lr=lr)
#opt  = torch.optim.SGD(model.parameters(),lr=lr,momentum=0.9)
opt  = torch.optim.SGD(model.parameters(),lr=lr,momentum=0.9,nesterov=True) 
optimizer_Description = str(opt)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=opt,mode='min',factor=0.1,patience=10,verbose=True)
```
## 9. Evaluation:
![Train vs Test Loss](https://github.com/sumankanukollu/cricket_scorePredict_in_first6_overs/raw/main/snippets/v2_test15.png)

From the abaove graph, we reached the global minima in 36th epoch, with a Train loss = 0.0061 & Test Loss = 0.0117
## 10. Scoring:
Lets see the predicted values of the model on test data:
```python
predicted_values = []
actual_values = []
for input,output in test_dl:
    for row in input:
        pred     = (loaded_model(torch.Tensor(row))*(encode_data['Total_score_max'] - encode_data['Total_score_min'] )) + encode_data['Total_score_min']
        predicted_values.append(pred)
    for op in output:
        actual    = (op*(encode_data['Total_score_max'] - encode_data['Total_score_min'] )) + encode_data['Total_score_min']
        actual_values.append(actual)
    break
```
```python
predicted_values[:5]
    [tensor([46.3414], grad_fn=<AddBackward0>),
    tensor([30.7063], grad_fn=<AddBackward0>),
    tensor([47.4587], grad_fn=<AddBackward0>),
    tensor([51.6460], grad_fn=<AddBackward0>),
    tensor([47.1634], grad_fn=<AddBackward0>)]

actual_values[:5]
    [tensor([49.]), 
    tensor([27.]), 
    tensor([64.]), 
    tensor([66.]), 
    tensor([64.])]
```

Based on these can say this model is 83.33% Accurate. Still there is a scope for improvement. Anyone can start from this point [access code from here](https://github.com/sumankanukollu/cricket_scorePredict_in_first6_overs/blob/main/IPL_Powerplay_Score_Prediction_v2.ipynb)