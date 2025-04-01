import pandas as pd
import plotly.graph_objects as go
import numpy as np
import matplotlib.pyplot as plt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Load the original dataset
df = pd.read_csv('emotion-emotion_updated.csv')



### Code for Class Mapping ###
# Map the existing emotion classes to four broader emotion classes
emotion_mapping = { 'sentimental': 'positive', 
                    'afraid': 'very negative', 
                    'proud':'very positive', 
                    'faithful': 'positive', 
                    'terrified': 'very negative', 
                    'joyful': 'very positive', 
                    'angry': 'very negative', 
                    'sad': 'negative', 
                    'jealous': 'negative', 
                    'grateful': 'very positive', 
                    'prepared': 'positive', 
                    'embarrassed': 'negative', 
                    'excited': 'very positive', 
                    'annoyed': 'negative', 
                    'lonely': 'negative', 
                    'ashamed': 'very negative', 
                    'guilty': 'very negative', 
                    'surprised': 'positive', 
                    'nostalgic': 'positive', 
                    'confident': 'very positive',
                    'furious': 'very negative', 
                    'disappointed': 'negative', 
                    'caring': 'positive', 
                    'trusting': 'very positive', 
                    'disgusted': 'very negative', 
                    'anticipating': 'positive', 
                    'anxious': 'negative', 
                    'hopeful': 'very positive', 
                    'content': 'positive', 
                    'impressed': 'very positive', 
                    'apprehensive': 'negative', 
                    'devastated': 'very negative'}

# Map the existing emotions to the new classes
df['updated_emotion'] = df['emotion'].map(emotion_mapping)

### Code for Univariate Analysis ###
updated_emotion_counts = {'very negative': 0, 'negative': 0, 'positive': 0, 'very positive': 0}
# Counts rows for each emotion
for emotion in updated_emotion_counts.keys():
    count = df[df['updated_emotion'] == emotion].shape[0]
    updated_emotion_counts[emotion] = count

# Analyze the mean and the standard deviation to detect outliers
emotion_counts = list(updated_emotion_counts.values())
mean_emotion_count = np.mean(emotion_counts)
sd_emotion_count = np.std(emotion_counts)
print("updated mean: " + str(mean_emotion_count) + ", updated standard deviation: " + str(sd_emotion_count))

# Plot the result
pdf = "eda_baseline.pdf"
emotions = []
counts = []
for emotion in updated_emotion_counts.keys():
    emotions.append(emotion)
    counts.append(updated_emotion_counts[emotion])
fig = go.Figure([go.Bar(x=emotions, y=counts)]) 
fig.write_image(pdf)

# Identify outliers whose counts are more than two standard deviations away from the mean
#threshold = 2
#outliers = []
#for emotion, count in emotions_dictionary_original.items():
#    z_score = (count - mean_emotion_count) / sd_emotion_count
#    if abs(z_score) > threshold:
#        outliers.append((emotion, count, z_score))



### Code for Text Length Analysis ###
# identify single-word and NaN inputs in the 'text' column. (may need further investigation for a cleaner dataset)
combined_df = df[df['text'].str.split().apply(len) == 1 | df['text'].isna()]
print("single-word and NaN inputs: \n" + str(combined_df))
# manually removed the rows that include single-number, character, and word 'text' inputs that do not make sense with the corresponding 'emotion' classes.



### Code for Bivariate Analysis ###
# 1. Evaluate each piece of text in sentiment scores using VADER. Question: what kind of sentiment score does each emotion tend to have?
analyzer = SentimentIntensityAnalyzer()

# 2. Add the 'sentiment_score' column that contains a compound sentiment score for each text. (Delete the column later)
df['sentiment_score'] = df['text'].apply(lambda x: analyzer.polarity_scores(x)['compound'])

# 3. Calculate mean sentiment scores for each emotion class
emotion_sentiment = df.groupby('updated_emotion')['sentiment_score'].mean()

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
fig.write_image('sentiment_scores_baseline.pdf')



### Outlier handling ###



# Update the csv file
df.to_csv('baseline.csv', index=False)

# References:
# ChatGPT for coding assistance
# Fine-Tuning DistilBERT for Emotion Classification: https://medium.com/@ahmettsdmr1312/fine-tuning-distilbert-for-emotion-classification-84a4e038e90e