import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk import tokenize
import csv
import matplotlib.pyplot as plt

data = pd.read_csv('reviews/reviews.csv')
Comments = data['comments'].head(1000)

sid = SentimentIntensityAnalyzer()

w = csv.writer(open("Analysed_Comments.csv", "w", encoding = 'utf-8', newline = ''))
count = 1
for comment in Comments:
	if count == 1:
		w.writerow(['Comment', 'pos', 'neg', 'neu'])
		count += 1
	else:
		ss = sid.polarity_scores(comment)
		values = []
		for key,value in ss.items():
			values.append((value))
			
		w.writerow([comment, values[2], values[0], values[1]])

