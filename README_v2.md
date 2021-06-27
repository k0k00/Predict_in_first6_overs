# IPL Powerplay Score Prediction in AWS Lambda
![IPL banner](https://cricketaddictor.gumlet.io/wp-content/uploads/2021/02/153757448_173638941016448_6980867142752435675_n.jpg?compress=true&quality=80&w=1920&dpr=2.6)


We start with a basic understanding powerplay in IPL, and quickly go through "Powerplay" score prediction --> based on this a predictive score in 20-overs.

## Introduction: 
It's COVID lockdown time. Everyone is worried about the increasing COVID cases, Work From Home (WFH) environment. No way to go out, no parties, no outing, no vacation plans, no gatherings ...... Government announces Lockdown : 1.0, 2.0, 3.0, 4.0 .....

At that time every cricket fan had a question. 
**"Do new Govt. rules in lockdown 4.0 pave the way for IPL 2020 ????"**



Finally IPL 2020 happend. 
Even pandamic situation can't derail the IPL juggernaut.

IPL 2020 earned revenue of : 4000 crores with 
*  35% reduced cost and 
*  25% increase in viewership. 

As a cricket fan I watch all the matches, during that time I observed that **"Powerplay plays a mojor role"** which is very Important in Teams score prediction.

## Powerplay in IPL:
"Powerplay" in IPL has fielding restrictions in 1<sup>st</sup> 6-overs, ie.. 


* only 2-fielders can stay outside the Inner-circle
  
  ![powerplay_field](https://github.com/sumankanukollu/cricket_scorePredict_in_first6_overs/raw/main/snippets/ppfield.JPG)
* After powerplay, upto 5-fielders can stay outside inner circle & 4-fielders must remain inside the inner circle.

    ![afterPowerplayField](https://github.com/sumankanukollu/cricket_scorePredict_in_first6_overs/raw/main/snippets/no_ppfield.JPG)
  
## Importance:
Powerplay makes the **batting compatively easy.**
Also **it's a trap for the batsmen**, as this will get them to take into a risk and loose their wickets in 1<sup>st</sup> 6-overs

## Effect of "Powerplay":
Powerplay overs are considered as pillars of any teams  victory.  
  **75% - of winning chance** depends on the Powerplay score. So, every team's expectation from the top 3-batsmen is      **"START THE INNINGS BIG"**





Steps of this blog are demonstrated in the below figure:

## Step-1: Load Dataset, Clean, Format and Transform:
* Source of the dataset : https://cricsheet.org/downloads/ipl_male_csv.zip
  | match_id | season  | start_date | venue                 | innings | ball | batting_team          | bowling_team                | striker     | non_striker | bowler  | runs_off_bat | extras | wides | noballs | byes | legbyes | penalty | wicket_type | player_dismissed | other_wicket_type | other_player_dismissed |
  | -------- | ------- | ---------- | --------------------- | ------- | ---- | --------------------- | --------------------------- | ----------- | ----------- | ------- | ------------ | ------ | ----- | ------- | ---- | ------- | ------- | ----------- | ---------------- | ----------------- | ---------------------- |
  | 335982   | 2007/08 | 4/18/2008  | M Chinnaswamy Stadium | 1       | 0.1  | Kolkata Knight Riders | Royal Challengers Bangalore | SC Ganguly  | BB McCullum | P Kumar | 0            | 1      |       |         |      | 1       |         |             |                  |                   |                        |
  | 335982   | 2007/08 | 4/18/2008  | M Chinnaswamy Stadium | 1       | 0.2  | Kolkata Knight Riders | Royal Challengers Bangalore | BB McCullum | SC Ganguly  | P Kumar | 0            | 0      |       |         |      |         |         |             |                  |                   |                        |
  | 335982   | 2007/08 | 4/18/2008  | M Chinnaswamy Stadium | 1       | 0.3  | Kolkata Knight Riders | Royal Challengers Bangalore | BB McCullum | SC Ganguly  | P Kumar | 0            | 1      | 1     |         |      |         |         |             |                  |                   |                        |
  | 335982   | 2007/08 | 4/18/2008  | M Chinnaswamy Stadium | 1       | 0.4  | Kolkata Knight Riders | Royal Challengers Bangalore | BB McCullum | SC Ganguly  | P Kumar | 0            | 0      |       |         |      |         |         |             |                  |                   |                        |

* Headers of this dataset, be self-explanatory. (so I am not explaing these)
* Next Identify the "Null" values in the dataset, hopefully there is no null values present in this dataset.
  

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



* ## Data Cleaning Steps:
  1. Here columns 'season' and 'start_date'  are not needed for our prediction. ```So we can drop the columns 'season' and 'start_date' from the dataset.```
   
  2. Delete Non-existing teams : 
      ```
      'Kochi Tuskers Kerala' 'Pune Warriors','Rising Pune Supergiants', 'Rising Pune Supergiant','Gujarat Lions'
      ```
  
  3. Replace the old team names with new team name:
      ```
      'Delhi Daredevils'  --> 'Delhi Capitals'
      'Deccan Chargers'   --> 'Sunrisers Hyderabad'
      'Punjab Kings'      --> 'Kings XI Punjab'
      ```

  4. Correct the venue column with unique names. In this dataset same stadium is being represented as in multiple ways. So identify those and rename.
      ```
      ['M Chinnaswamy Stadium', 'M.Chinnaswamy Stadium']
      ['Brabourne Stadium', 'Brabourne Stadium, Mumbai']
      ['Punjab Cricket Association Stadium, Mohali', 'Punjab Cricket Association IS Bindra Stadium, Mohali', 'Punjab Cricket Association IS Bindra Stadium']
      ['Wankhede Stadium', 'Wankhede Stadium, Mumbai']
      ['Rajiv Gandhi International Stadium, Uppal', 'Rajiv Gandhi International Stadium']
      ['MA Chidambaram Stadium, Chepauk','MA Chidambaram Stadium',      'MA Chidambaram Stadium, Chepauk, Chennai']
      ```

  5. Rename the column names:
        ```
        'striker'     --> 'batsmen'
        'non-striker' --> 'batsmen_nonstriker'  (This column is not required)
        'bowler'      --> 'bowlers'
        ```
      
  6. Create a columns "Total_score" : which reflects the runs through bat and extra runs through wides,byes,noballs,legbyes...etc.  
    Hence we can drop columsn 
    ```['wides', 'noballs', 'byes', 'legbyes', 'penalty', 'wicket_type','other_wicket_type', 'other_player_dismissed']```
  7. **Select required Data:**
   
      Here as we are preedicting the "powerplay" score we can drop rest of the data, 
                
          * Select 1st 6-Overs details  and  
          * Drop all innings > 2 
              As there is no scope for 3rd and 4th innings
* **Data Transform:**
  
  1. We can see the cleaned data is having multiple Dtype's (like int64 and float64 and object).
     1. so, we should Transform dataframe into single Dtype using "sklearn label encoder" 
  2. Here we can see, few players who can bat as well as bowl.
    Means same player will be listed as a batsmen and as well as bowler.
       * So to make the prediction properly, I am creating one Dataframe with all the players name it as "players_df", which I use for encoding the players with some value to identify.
* **Format the Dataset:**
  
  * Club the all rows with respect to matchID and Innings.
  * Grab all the batsmen and bowler details who batted and bowled in 1<sup>st</sup> 6-overs
  * Calculate the total score (runs through bat + extra runs)
  * How many players dismissed in 1st 6-overs (will be used as another parameter for prediction)
  * Align the batsmen and bowlers details in to a separate column

    In above formated dataset, we got list of batsmen and bowlers details who batted and bowled in 6-overs.

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
* **Transform the dataset:**
    * As part of this transformation, encode all the object values with the encoded values, **make sure player name or any object should be unique for the entire dataframe where ever it repeat**   
  
# Finally dataset will be transformed like below:
| venue | innings | batting_team | bowling_team | bat1 | bat2 | bat3 | bat4 | bat5 | bat6 | ... | bow1 | bow2 | bow3 | bow4 | bow5 | bow6 | runs_off_bat | extras | Total_score | player_dismissed |   |
| ----- | ------- | ------------ | ------------ | ---- | ---- | ---- | ---- | ---- | ---- | --- | ---- | ---- | ---- | ---- | ---- | ---- | ------------ | ------ | ----------- | ---------------- | - |
| 0     | 15      | 1            | 3            | 6    | 419  | 70   | 385  | 0.0  | 0.0  | 0.0 | ...  | 326  | 512  | 19   | 0.0  | 0.0  | 0.0          | 51     | 10          | 61               | 1 |
| 1     | 15      | 2            | 6            | 3    | 351  | 496  | 483  | 187  | 97   | 297 | ...  | 22   | 163  | 20   | 0.0  | 0.0  | 0.0          | 19     | 7           | 26               | 4 |
| 2     | 23      | 1            | 0            | 2    | 331  | 286  | 274  | 0.0  | 0.0  | 0.0 | ...  | 65   | 408  | 201  | 0.0  | 0.0  | 0.0          | 50     | 3           | 53               | 1 |
| 3     | 23      | 2            | 2            | 0    | 207  | 201  | 0.0  | 0.0  | 0.0  | 0.0 | ...  | 184  | 296  | 0.0  | 0.0  | 0.0  | 0.0          | 61     | 2           | 63               | 1 |
| 4     | 10      | 1            | 5            | 1    | 468  | 506  | 443  | 255  | 0.0  | 0.0 | ...  | 146  | 62   | 275  | 0.0  | 0.0  | 0.0          | 38     | 2           | 40               | 2 |

* 

  

  









