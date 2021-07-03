# nfl_kickers_analysis

- Check out the medium post about this project: https://medium.com/@danieleliezer/nfl-kicker-analysis-a57e023088c1

- The data was collected from this kaggle dataset: https://www.kaggle.com/toddsteussie/nfl-play-statistics-dataset-2004-to-present. The data is from 2006 to 2019.

Files in the repository:

• plays.csv: A full play-by-play report, with the information about the play, such as the attacking team, the defending team, the score before and after, the distance to the goal post, the moment on the game, the outcome of the play, etc. 

• games.csv: A report with the information about the games, such as the season, week, stadium, score, teams playing, etc.
<br>• players.csv: A report with the information about the players: such as the first name, last name, nationality, date of birth, age of draft, college, etc.
<br>• nfl_kicker_analysis.ipynb: The notebook of the project
<br>• nfl_kicker_analysis.py: The project in python

Libraries used:
Pandas, Numpy, Seaborn, Matplotlib, Scipy, dateutil, datetime

Description and Motivation

The kickers job is a hard one. In this project, the goal is to do an EDA to better understand the performance of NFL kickers. 

We'll try to answer some of the followings: 

• What are the best and worst teams in the NFL kicking the ball? And the best players? <br>
• What is the probability of scoring as a function of goal distance? What's the average accuracy at the "field goal range"? <br>
• Are the kickers improving their accuracy over the years? <br>
• Do the kickers feel the pressure? How do their accuracy change when playing home vs away? And in pivotal moments? <br>
• At what ages do the kickers hit their prime? 

- Main Results

• The Patriots, the Ravens and the Steelers were great kicking the ball in the period. The Browns, Buccaneers and the Washington Football Team were the worst. <br>
• Justin Tucker stood out a lot! <br>
• In the "field goal range" limit, the average accuracy is 63% <br>
• The league have better player over the seasons! <br>
• The kickers do feel the pressure in pivotal moments! Their accuracy are considerably worse. <br>
• We can't be sure about the age that the kickers are in their prime. We would need a longer period to make this study properly.


Thanks for reading!
