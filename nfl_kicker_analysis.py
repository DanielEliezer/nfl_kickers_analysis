#!/usr/bin/env python
# coding: utf-8

# # NFL Kicker Analysis
# ***

# ![image-2.png](attachment:image-2.png)
# 
# Welcome to the NFL kicker’s performance analysis! We’ll take a dive on the data from kickers from 2006 and 2019. Are they getting better over the years? Do they feel the pressure of playing away from home? How do their accuracy change in pivotal moments? We'll try to get some of these answers.
# 
# The data was obtained in <a href="https://www.kaggle.com/toddsteussie/nfl-play-statistics-dataset-2004-to-present"> this kaggle dataset. </a>
# It contains a full play-by-play report of all games in the NFL. It’s a very big folder with a lot of csv files that can generate a lot of interesting analysis for the football fans. 
# 
# #### In this project, we’ll try to answer the followings:
# <br>1) What are the best and worst teams in the NFL kicking the ball? And the best players?
# <br>2) What is the accuracy of the kicks as a function of goal distance? What’s the average accuracy at the “field goal range”?
# <br>3) Are the kickers improving their accuracy over the years?
# <br>4) Do the kickers feel the pressure? How do their accuracy change when playing home vs away? And in pivotal moments?
# <br>5) In what ages can we expect the kicker to be in his prime?
# <br>	
# 	Ready? Let’s go!
# 
# ***

# In[1]:


## Imports

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from dateutil.relativedelta import relativedelta
from datetime import timedelta
from scipy.stats import chisquare

## Dictionary to convert team id to name

dict_teams = {
4500 : 'San Francisco 49ers', 3300 : 'New Orleans Saints', 4600 : 'Seattle Seahawks', 3000 : 'Minnesota Vikings',
2310 : 'Kansas City Chiefs', 610 : 'Buffalo Bills', 2120 : 'Houston Texans', 4900 : 'Tampa Bay Buccaneers',
1200 : 'Dallas Cowboys', 3800 : 'Arizona Cardinals', 2250 : 'Jacksonville Jaguars', 3200 : 'New England Patriots',
1800 : 'Green Bay Packers', 325 : 'Baltimore Ravens', 1050 : 'Cleveland Browns', 3410 : 'New York Giants',
2700 : 'Miami Dolphins', 2520 : 'Las Vegas Raiders', 3700 : 'Philadelphia Eagles', 2100 : 'Tennessee Titans',    
810 : 'Chicago Bears', 3430 : 'New York Jets', 200  : 'Atlanta Falcons', 2510 : 'Los Angeles Rams',    
920 : 'Cincinnati Bengals', 4400 : 'Los Angeles Chargers', 2200 : 'Indianapolis Colts', 750 : 'Carolina Panthers',
3900 : 'Pittsburgh Steelers', 5110 : 'Washington Football Team', 1540 : 'Detroit Lions', 1400 : 'Denver Broncos'}


## defining functions

def fill_na_values(df):
    """
    Fill the null fields with the correct values
    """
    list_aux = []
    for i in range(len(df['season'])):

        if (pd.isna(df.loc[i, 'distance'])) & (df.loc[i, 'season'] < 2015):
            list_aux.append(20)
        elif (pd.isna(df.loc[i, 'distance'])) & (df.loc[i, 'season'] >= 2015):
            list_aux.append(33)  
        else:
            list_aux.append(df.loc[i, 'distance'])
        
    df['distance'] = list_aux
    
    
def get_kicker_name(st):
    
    """
    Extracts the kicker name from the play description
    """
    x = st[st.find("(")+1:st.find(")")]
    x = x[:1]+'.'+ x[x.find(".")+1:]
    return x 
    


# ## Creating the main dataset
# ***
# 
# We are going to create the main dataset to answer all of our questions. The data comes from the play-by-play file, and the games file. 

# In[2]:


## ps: I had to make some changes in this file, so that I could upload on github. The original dataset i
plays = pd.read_csv('plays.csv')
plays.head()


# In[3]:


games = pd.read_csv('games.csv')
games.head()


# In[4]:


## Filtering the raw datasets, and selecting only the most important columns  
plays = plays[['playId','gameId','possessionTeamId','nonpossessionTeamId','playType',
               'quarter','gameClock','playStats', 'distanceToGoalPre','netYards', 'offensiveYards',
               'visitingScorePost','visitingScorePre','homeScorePost','homeScorePre']]
plays = plays[plays['playType'].isin(['field goal','xp'])]

games = games[games['seasonType'] == 'REG']
games = games[['season', 'gameId', 'gameDate','homeTeamId','visitorTeamId']]

# filtering and joining the data
df = plays.merge(games[['season', 'gameId', 'gameDate','homeTeamId','visitorTeamId']], on = 'gameId', how = 'inner')

# Create useful columns to make calculations easier

df['points_scored'] = df['visitingScorePost'] - df['visitingScorePre'] + df['homeScorePost'] - df['homeScorePre']

df['scored'] = df['points_scored'] != 0

df.rename({'distanceToGoalPre':'distance'}, axis = 1, inplace = True)

df.loc[df['playType'] == 'xp' ,'potential_pts'] = 1
df.loc[df['playType'] == 'field goal' ,'potential_pts'] = 3

df['home_team_kick'] = df['homeTeamId']==df['possessionTeamId']
df.loc[df['home_team_kick'] == True ,'home_team_kick'] = 'Home'
df.loc[df['home_team_kick'] == False ,'home_team_kick'] = 'Away'

df['possession_team_name'] = df['possessionTeamId'].map(dict_teams)
df['nonpossession_team_name'] = df['nonpossessionTeamId'].map(dict_teams)
df['home_team_name'] = df['homeTeamId'].map(dict_teams)

df = df.drop(['possessionTeamId','nonpossessionTeamId','gameClock','netYards','offensiveYards', 'visitingScorePost','homeScorePost','homeTeamId','visitorTeamId'], axis = 1)

display(df.head())
display(df.shape)


# ## Data Cleaning
# ***
# 
# Like every data science project, we must check for inconsistencies in the dataset and fix them.  

# In[5]:


df['points_scored'].value_counts()


# We shouldn't consider the rows with 2 or 6 points, as they didn't happen due to the action of a kicker. Let's drop them.

# In[6]:


df = df[df['points_scored'].isin([0,1,3])].reset_index(drop = True)
df


# Checking at the values of “distance”, this number is the distance between the beginning of the play to the end zone. What we actually want it to show, is where the kick hits the ball. Therefore, we need to add 17 yards to this number, and 18 in case of extra points.

# In[7]:


#df['distance'] = df['distance']+18
df.loc[df['playType'] == 'xp' ,'distance'] = df['distance']+18
df.loc[df['playType'] == 'field goal' ,'distance'] = df['distance']+17

## Checking for nulls
df.isnull().sum()


# For some reason, some data weren't filled correctly. Luckily, all of them are related to extra points, and these numbers can be easily filled: Before 2015, the distance of the field goal was 20 yards. After that, they changed it to 33. Let’ fix it.

# In[8]:


fill_na_values(df)

## Checking for nulls
df.isnull().sum()


# Now that we have a clean dataset, we can start getting some answers.

# ## What are the best/worst teams in the NFL kicking the ball? And how about the players?
# ***
# 
# 
# ### Best Teams
# To determine how good is a certain group of kickers, we measure their accuracy (kicks scored/total kicks). We won't make any distinctions between field goals and extra points.

# In[9]:


## create temporary dataframe to make the chart
dummy_df = pd.DataFrame(df.groupby(['possession_team_name']).mean()['scored'].sort_values(ascending = False)).reset_index() 
dummy_df['scored'] = dummy_df['scored']*100

fig = plt.figure(figsize = (12,15))
plt.title("NFL TEAMS ACCURACY (%)", fontweight = 'bold', fontsize = 17)
ax = sns.barplot(x = 'scored', y = 'possession_team_name', data = dummy_df, palette = 'coolwarm')
plt.xlabel("Score %", fontsize = 14)
plt.ylabel("", fontsize = 14)

for index, value in enumerate(dummy_df["scored"].round(1)):
    plt.text(value, index, str(value), color='dimgray', fontweight="bold", fontsize=12, verticalalignment='center')


# The Patriots, Ravens and Steelers have the best score percentage in these years. The Buccaneers, Washington and Browns have the worst.
# 
# Similarly, let's see what the best players are. Since the extra point distance changed in 2015, we might want to remove them from this analysis, as they can mess up the comparison.

# In[10]:


## Make a regex to gatter the kicker name
df['kicker_name'] = df['playStats'].apply(get_kicker_name)
df['count_kicker'] = df.groupby("kicker_name")['kicker_name'].transform('count')

## Create a dummy df, filtering kickers with at least 200 kicks
dummy_df = pd.DataFrame(df.query("count_kicker > 200 and playType == 'field goal'").groupby(['kicker_name']).mean()['scored'].sort_values(ascending = False)).reset_index() 
dummy_df['scored'] = dummy_df['scored']*100

fig = plt.figure(figsize = (12,10))
plt.title("NFL PLAYERS ACCURACY (%)", fontweight = 'bold', fontsize = 17)
ax = sns.barplot(x = 'scored', y = 'kicker_name', data = dummy_df[:15], color ="tab:green");
plt.ylabel('', fontsize = 13)
plt.xlabel('Accuracy (%)', fontsize = 13)

for index, value in enumerate(dummy_df["scored"][:15].round(1)):
    plt.text(value, index, str(value), color='dimgray', fontweight="bold", fontsize=12, verticalalignment='center')


# How does this compare to the rest of the league?

# In[11]:


fig = plt.figure(figsize = (10,6))
plt.title("PLAYERS ACCURACY (%)", fontsize = 16, fontweight = 'bold')
ax = sns.boxplot(x = dummy_df['scored'], color = 'tab:red')
plt.xlabel("Accuracy (%)", fontsize = 14);


# We can see that Justin Tucker stands out a lot!

# ## What is the accuracy of the kicks as a function of goal distance?
# ***

# In[12]:


## creating dummy dataset to facilitate the plot
df_dummy = df.groupby("distance").mean()[['scored']].reset_index()
df_dummy['scored'] = df_dummy['scored']*100

## plotting
fig = plt.subplots(figsize=(12,7))
plt.title("ACCURACY VS DISTANCE", fontweight = 'bold', fontsize = 15)
ax = sns.lineplot(data = df_dummy, x = 'distance', y = 'scored', color = 'tab:blue')
plt.axvspan(0, 52, facecolor='0.8', alpha=0.5, label="Field Goal Range")
plt.xlabel("Aproximate Distance (yd)", fontsize = 14)
plt.ylabel("Accuracy (%)", fontsize = 14)
plt.xlim(17,64)
plt.grid(linestyle = '--')
plt.legend()
plt.show();


# We can see that the accuracy in the field goal range limit (52 yd) is 63%. After that, it falls very quickly. For example: at 55 yards, the accuracy is 55%, after 1 yard, the accuracy is 45%.
# 
# 

# ## Are the kickers improving over the years?
# ***
# 
# First of all, let’s plot the average accuracy over the years. Again, we will remove extra points from this analysis.

# In[13]:


# creating auxiliar dataset
dummy_df = pd.DataFrame(df[df['playType']=='field goal'].groupby("season").mean()['scored']*100).reset_index()

# plotting
plt.figure(figsize = (12,7))
plt.title("ACCURACY OVER THE SEASONS", fontsize = 16, fontweight = 'bold')
sns.barplot(data = dummy_df, x = 'season', y = 'scored', color = "tab:green")
plt.axhline(y = df[df['playType']=='field goal']['scored'].mean()*100, color = 'gray', linestyle = '-', label = 'mean')
plt.ylim(50,100)
plt.xlabel('Season', fontsize = 14)
plt.ylabel('Accuracy (%)', fontsize = 14)
plt.legend();


# The accuracy is on the rise! Before jumping into any conclusion, let’s make sure that the reason for this increase is due to a better performance of players, and not a drop in distances.

# In[14]:


## Create a temporary dataset to make calculations easier 
df['approximate_distance'] = df['distance']//10*10
_ = df[df['playType']=='field goal']
dummy_df = pd.DataFrame(_.groupby(["season"]).mean()['distance']).reset_index()

## plotting
plt.figure(figsize = (12,7))
sns.barplot(data = dummy_df, x = 'season', y = 'distance', color = 'tab:blue')
plt.title("AVERAGE DISTANCE VS SEASON ", fontsize = 16, fontweight = 'bold')
plt.ylim(0,45)
plt.axhline(y = _['distance'].mean(), color = 'gray', linestyle = '-', label = 'mean')
plt.xlabel('Season', fontsize = 14)
plt.ylabel('Distance (yd)', fontsize = 14)
plt.legend();


# That settles it. The distance of the field goals are also increasing, we can confirm that **the kickers are getting better over the seasons.**

# ## Do the kickers feel the pressure? How do their accuracy change when playing away? How about in pivotal moments?
# ***
# 
# 
# 
# ### Home vs Away
# First, let's compare the accuracy of the kicker playing home vs away.

# In[15]:


df.groupby(['home_team_kick']).mean()[['scored','distance']]


# Looking at the overall number won't help us a lot. The accuracy at home is higher, but we are not sure that this is due to the performance of the kickers, or the distance, that is slightly smaller. We should break this data into ranges of distance to make this comparassion.

# In[16]:


## To display fewer data, we'll create a new category that group similar distances.
## Ex: 0 to 9 yd => 0 yd; 10 to 19 yd => 10 yd ... 

df['approximate_distance'] = df['distance']//10*10
df_dummy = df.groupby(['home_team_kick','approximate_distance'])['scored'].mean().reset_index()
df_dummy['scored'] = df_dummy['scored']*100 

fig = plt.figure(figsize = (10,6))
sns.barplot(data = df_dummy, x = 'approximate_distance', y = 'scored', hue = 'home_team_kick');
plt.title("ACCURACY AT HOME VS AWAY", fontsize = 16, fontweight = 'bold')
plt.xlabel("Distance (yd)", fontsize = 14)
plt.ylabel("Accuracy (%)", fontsize = 14)
plt.legend(loc = 1);


# Visualizing the data, it's hard to jump into any conclusion. For most ranges, it seems that the home team kicker is more accurate. But that is not true for all of them: from 10 to 20 yards and 40 to 50, the away kicker is slightly more accurate. These oscillations don't make much sense if there were a big advantage on kickers playing at home, as we might expect. Therefore, we can't assume that there's a big advantage at playing at home.

# ### Pivotal Moments
# 
# First of all, let's define pivotal moments as the kicks in the second half of the game that can change the lead or tie the match. In these situations, we can expect that the pressure on the kicker is much higher. Does that affects a lot their performance?

# In[17]:


## Creating a "pivotal moment column", that indicates if that kick qualifies as a pivotal moment.

df['pivotal_moment'] = False
df.loc[(df['quarter']>=3)&(df['home_team_kick']=='Home')&((df['homeScorePre'] - df['visitingScorePre'] + df['potential_pts']).isin([0,1,2,3])), 'pivotal_moment'] = True
df.loc[(df['quarter']>=3)&(df['home_team_kick']=='Away')&((df['visitingScorePre'] - df['homeScorePre'] + df['potential_pts']).isin([0,1,2,3])), 'pivotal_moment'] = True

## Plotting the mean values
_ = df.groupby(['pivotal_moment']).mean()[['scored','distance']]
_['scored'] = _['scored']*100
display(_)


# Again, it's hard to jump into a conclusion. Checking the overall number, the accuracy in these moments drops a lot, but the distance is also bigger. We must make these comparisons individually for each interval

# In[18]:


## Create a dummy dataset
df_dummy = df.groupby(['pivotal_moment','approximate_distance'])['scored'].mean().reset_index()
df_dummy['scored'] = df_dummy['scored']*100 

## Plotting
plt.figure(figsize = (12,7))
plt.title("DIFFERENCE OF ACCURACY IN HIGH PRESSURE SITUATIONS", fontsize = 16, fontweight = 'bold')
sns.barplot(data = df_dummy, x = 'approximate_distance', y = 'scored', hue = 'pivotal_moment')
plt.legend(title="Pivotal Moment",loc=1, fontsize='medium', fancybox=True)
plt.xlabel("Distance (yd)", fontsize = 14)
plt.ylabel("Accuracy (%)", fontsize = 14);


# For almost every interval, the kickers in pivotal moments have a worse performance, indicating that they are not as good under pressure. For the ranges of 10-20 and 60-70, the opposite happened. Since those are the ranges with the smallest sample size, maybe that could be interfering in the results. 
# 
# Before running a chi-squared test for these intervals, we'll need to check if we have the minimum sample size in all groups.

# In[19]:


## Plotting count of kicks in pivotal situations for each interval
display(df[df['pivotal_moment']==True].groupby(['pivotal_moment','approximate_distance']).count()['playId'].reset_index().rename({"playId":"count"}, axis = 1)[['approximate_distance','count']])


# According to  <a href="https://www.statskingdom.com/sample_size_chi2.html"> this calculator</a>, the sample size should be **at least 88**. Therefore, we'll run the tests for the kicks between 20 and 59 yards.
# 

# In[20]:


## Running the chi-squared test for the ranges in the list
ranges = [20,30,40,50]

print("Results of the chi-squared test:")
print("")
for dist in ranges:
    df_dummy = df[df['approximate_distance'] == dist]
    contigency= pd.crosstab(df_dummy['pivotal_moment'], df_dummy['scored']) 
    chi_2, p_valor = chisquare(f_obs=contigency.loc[True,:], f_exp=contigency.loc[False,:])
    print("For {} yards, the p_value is {}.".format(dist,round(p_valor,4)))


# For every range, the p value is under the significance (0.05). Thus, we have a statistical guarantee that **players are worse under high pressure situations.**

# ## At what ages the kicker hits his prime?
# ***
# This question is interesting: Since the kickers don’t have the same work rate as the other positions, it’s fair to assume that they can play until an older age. Not only that, their job depends a lot on good mental preparation. More experienced players could have an advantage on that.
# 
# As we don't have the age of the kicker in the current dataset, we must gatter this information. It can be found at the players file

# In[21]:


## Importing player data

players = pd.read_csv('players.csv')
players = players[players['position'] == 'K']
players['kicker_name'] = players['nameFirst'].str[:1]+'.'+players['nameLast']


# We have an issue here: We don't have a player_id on the previous df. Therefore, this join will be made using the player name as the key... we can expect that this join match won't be flawless, since we got the player name through a regex.

# In[22]:


# Merging the player data to the new dataset
df_merged = df.merge(players, how = 'inner', on = 'kicker_name')

# Comparing the lenght of the original dataset, to the new one, and the number of unique play_ids
df.shape[0],df_merged.shape[0], df_merged.playId.nunique()


# Two considerations: 
# <br> • Between the original dataset and the new one, we lost a little over 1000 records. That's not a big deal, we still have plenty of data to our analysis.
# <br> • We can see that the new dataset have some duplicates. Mostly that's because some names are very similar to the other, and we can't tell them apart. e.g: 'Josh Brown' and 'Jon Brown'. Since it's a very small number, we'll drop them.

# In[23]:


## Dropping duplicated dat
df_merged = df_merged.drop_duplicates(subset=['playId'], keep=False)

## Checking if the new dataframe have duplicated kicks
assert df_merged.shape[0] == df_merged.playId.nunique()


# In[24]:


## getting the age of the players
df_merged['gameDate'] = pd.to_datetime(df_merged['gameDate'])
df_merged['dob'] = pd.to_datetime(df_merged['dob'])
df_merged['age'] = (df_merged['gameDate'] - df_merged['dob'])/timedelta(days=365)//1
df_merged.head()


# In[25]:


## Creating a dummy dataset
dummy_df = df_merged.groupby(['age'])['scored'].agg(['count', 'mean']).reset_index()
dummy_df['mean'] = dummy_df['mean'] * 100

## Plotting the data
plt.figure(figsize = (12,7))
sns.lineplot(data = dummy_df, x = 'age', y = 'mean', color = 'darkred')
plt.title("ACCURACY VS PLAYER AGE", fontweight = 'bold', fontsize = 16)
plt.xlabel("Age", fontsize = 14)
plt.ylabel("Accuracy (%)", fontsize = 14)
plt.axhline(y = dummy_df['mean'].mean(), color = 'gray', linestyle = '--')
plt.ylim(80,102);


# Looking at the chart, we can see that the graph has a growing trend, and before the age of 31, the accuracy over the years fluctuates around the average. After that, it kept increasing until the early 40’s.
# 
# Based on this data, are we 100% certain that older kickers are better than younger ones? 
# 
# #### No! We probably have a case of survivorship bias. 
# 
# "The survivorship bias is the logical error of concentrating on the people or things that made it past some selection process and overlooking those that did not, typically because of their lack of visibility"
# 
# In our case, as the kickers get older, if they don’t have enough game time, they may decide to retire. That means that, older players who have not retired yet tend to be very good, since they still have a lot of gametime.
# 

# In[26]:


dummy_df = df_merged.groupby(['age'])['scored'].agg(['count', 'mean']).reset_index()
dummy_df['mean'] = dummy_df['mean'] * 100

plt.figure(figsize = (10,6))
sns.lineplot(data = dummy_df, x = 'age', y = 'count', color = 'tab:blue')
plt.title("NUMBER OF KICKS VS AGE OF THE KICKER", fontweight = 'bold', fontsize = 16)
plt.plot();


# Looking at the count of kicks by age, we can see that this number takes a huge drop after 35 years. This strengthens the idea that the worst players retire early, or get benched.
# 
# Therefore, we can’t be sure if older kicker are better than younger ones or not. This analysis is not easy to do and it would require longer periods of data. With an appropriate dataset, we could see for example, for every retired player, what age was his prime. Having only 14 years of data, that's not possible.
