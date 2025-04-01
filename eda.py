import pandas as pd
import plotly.graph_objects as go
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Load the original dataset
df = pd.read_csv('emotion-emotion_69k.csv')



### Code for Univariate Analysis ###
# Dictionary that will be used for counting data points by emotion class
emotions_dictionary_original = {'sentimental':0, 'afraid':0, 'proud':0, 'faithful':0, 'terrified':0, 'joyful':0, 'angry':0, 'sad':0, 'jealous':0, 'grateful':0, 
                                'prepared':0, 'embarrassed':0, 'excited':0, 'annoyed':0, 'lonely':0, 'ashamed':0, 'guilty':0, 'surprised':0, 'nostalgic':0, 'confident':0,
                                'furious':0, 'disappointed':0, 'caring':0, 'trusting':0, 'disgusted':0, 'anticipating':0, 'anxious':0, 'hopeful':0, 'content':0, 
                                'impressed':0, 'apprehensive':0, 'devastated':0}

# Counts rows for each emotion
for emotion in emotions_dictionary_original.keys():
    count = df[df['emotion'] == emotion].shape[0]
    emotions_dictionary_original[emotion] = count

# Analyze the mean and the standard deviation to detect outliers
emotion_counts = list(emotions_dictionary_original.values())
mean_emotion_count = np.mean(emotion_counts)
sd_emotion_count = np.std(emotion_counts)

# Identify outliers whose counts are more than two standard deviations away from the mean
threshold = 2
outliers = []
for emotion, count in emotions_dictionary_original.items():
    z_score = (count - mean_emotion_count) / sd_emotion_count
    if abs(z_score) > threshold:
        outliers.append((emotion, count, z_score))

# Plot the result
pdf = "eda_original.pdf"
emotions = []
counts = []
for emotion in emotions_dictionary_original.keys():
    emotions.append(emotion)
    counts.append(emotions_dictionary_original[emotion])
fig = go.Figure([go.Bar(x=emotions, y=counts)]) 
fig.write_image(pdf)



### Code for Bivariate Analysis ###
# 1. Evaluate each piece of text in sentiment scores using VADER. Question: what kind of sentiment score does each emotion tend to have?
analyzer = SentimentIntensityAnalyzer()

# 2. Add the 'sentiment_score' column that contains a compound sentiment score for each text. (Delete the column later)
df['sentiment_score'] = df['situation'].apply(lambda x: analyzer.polarity_scores(x)['compound'])

# 3. Calculate mean sentiment scores for each emotion class
emotion_sentiment = df.groupby('emotion')['sentiment_score'].mean()

# 4. Plot the result
emotions = emotion_sentiment.index
sentiment_scores = emotion_sentiment.values
fig = go.Figure()
fig.add_trace(go.Bar(
    x=emotions,
    y=sentiment_scores,
    name='Sentiment score',
    marker_color='lightblue'
))
fig.add_trace(go.Scatter(
    x=emotions,
    y=[0] * len(emotions),
    mode='lines',
    line=dict(color='black', dash='dash'),
    name='Neutral Sentiment'
))
fig.update_layout(
    title='Sentiment Scores for Each Emotion',
    xaxis_title='Emotion',
    yaxis_title='Sentiment Score',
    showlegend=False,
    xaxis=dict(showline=False, showgrid=False),
    yaxis=dict(showline=False, showgrid=False, zeroline=False), 
    plot_bgcolor='white', 
    margin=dict(l=0, r=0, t=40, b=60)
)
fig.write_image('sentiment_scores_plot.pdf')



### Outlier handling ###
# Exclude 'faithful', 'surprised', and 'jealous'
df = df[~df['emotion'].isin(['faithful', 'surprised', 'jealous'])]
df = df.drop(columns=['sentiment_score'])



# Update the csv file
df.to_csv('eda.csv', index=False)

# References:
# ChatGPT for coding assistance