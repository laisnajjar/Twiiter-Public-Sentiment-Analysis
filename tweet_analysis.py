"""Tweet Analysis"""
import os
import shutil
import json
import glob
import re
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import pipeline
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk import FreqDist
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
# Download required NLTK data
nltk.download('vader_lexicon', quiet=True)
nltk.download('stopwords')
# ------------------------------------ Configuration ------------------------------------

COUNTRY = 'Palestine'  # Change to 'Israel' for Israel analysis
FILE_PATTERN = 'pto*.json' if COUNTRY == 'Palestine' else 'ito*.json'
BASE_PATH = f'Collected_Tweets/{COUNTRY}/v2'
STOPWORDS_FILE = "gist_stopwords.txt"
OUTPUT_DIR = f'Output/{COUNTRY}'
stops = set(stopwords.words('english'))
# ------------------------------------ Utility Functions ------------------------------------
def clean_directory(directory):
    """Remove all files from a given directory."""
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')

def read_json_files(base_path, pattern):
    """Read and extract text from JSON files based on pattern."""
    tweets = []
    full_pattern = f'{base_path}/{pattern}'
    for file in glob.glob(full_pattern):
        with open(file, 'r') as f:
            print(f'Reading {file}')
            try:
                data = json.load(f)
                tweets.extend(tweet['text'] for tweet in data['data'])
                with open("Eval", 'a', encoding="UTF-8") as file:
                    json.dump(tweets, file, indent=4, sort_keys=True)
            except json.decoder.JSONDecodeError:
                print(f'Error reading {file}.')
                continue
    return tweets

def clean_and_tokenize(text, stopwords_set):
    """Clean and tokenize text."""
    url_re = re.compile(r'http\S+')
    non_alpha_re = re.compile(r'[^a-zA-Z\s]')
    text = url_re.sub('', text)
    text = non_alpha_re.sub('', text)
    text = text.lower()
    # text = [token for token in text.split() if token not in stopwords_set]
    tokens = word_tokenize(text)
    # Remove stopwords
    tokens = [token for token in tokens if token not in stopwords_set]
    return tokens

def load_stopwords(file_path):
    """Load stopwords from a file."""
    with open(file_path, "r") as file:
        return set(file.read().split(","))

def categorize_sentiment(score):
    """Categorize sentiment score."""
    if score > 0.05:
        return 'positive'
    elif score < -0.05:
        return 'negative'
    else:
        return 'neutral'

emotion_classifier = pipeline(
    'text-classification',
    model='bhadresh-savani/distilbert-base-uncased-emotion'
)
def detect_emotions(texts):
    """Detect emotions in a list of texts."""
    return emotion_classifier(texts)
# # ------------------------------------ Main Analysis ------------------------------------
os.makedirs(OUTPUT_DIR, exist_ok=True)
clean_directory(OUTPUT_DIR)
# Load stopwords
stopwords = load_stopwords(STOPWORDS_FILE)

# Read tweets
tweets = read_json_files(BASE_PATH, FILE_PATTERN)
#cleaned_tweets = [' '.join(clean_and_tokenize(tweet, stopwords)) for tweet in tweets]
cleaned_tweets = [' '.join(clean_and_tokenize(tweet, stops)) for tweet in tweets]
print(f'Number of Tweets: {len(tweets)}')
# Sentiment Analysis
sia = SentimentIntensityAnalyzer()
sentiments = [sia.polarity_scores(tweet)['compound'] for tweet in cleaned_tweets]
average_sentiment = np.mean(sentiments)
print(f"Average Sentiment Score for {COUNTRY}: {average_sentiment}")


# TF-IDF and Cosine Similarity
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(cleaned_tweets)

# Word Cloud Generation
word_indices = [i for i, tweet in enumerate(cleaned_tweets) if COUNTRY.lower() in tweet]
ranked_words = defaultdict(int)
for index in word_indices:
    feature_array = vectorizer.get_feature_names_out()
    tfidf_sorting = np.argsort(tfidf_matrix[index].toarray()).flatten()[::-1]
    top_n = feature_array[tfidf_sorting][:10]
    for word in top_n:
        ranked_words[word] += 1

wordcloud = WordCloud(
    width=800,
    height=400,
    background_color='white').generate_from_frequencies(ranked_words)

# Plotting
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title(f"Word Cloud for {COUNTRY} based on TF-IDF")
plt.savefig(os.path.join(OUTPUT_DIR, f'{COUNTRY}_TFIDF_WordCloud.png'))

# Create a frequency distribution of all the tokens
all_words = ' '.join(cleaned_tweets).split()
frequency_distribution = FreqDist(all_words)

# Generate Word Cloud
wordcloud = WordCloud(
    width=800,
    height=400,
    background_color='white'
).generate_from_frequencies(frequency_distribution)

# Plotting
plt.figure(figsize=(15, 7))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title(f"Word Cloud for {COUNTRY} based on Term Frequency")
plt.savefig(os.path.join(OUTPUT_DIR, f'{COUNTRY}_TF_WordCloud.png'))

# Sentiment Scatter Plot
sentiment_categories = [categorize_sentiment(score) for score in sentiments]
plt.figure(figsize=(12, 6))
for category in ['positive', 'negative', 'neutral']:
    indices = [i for i, cat in enumerate(sentiment_categories) if cat == category]
    plt.scatter(indices, [sentiments[i] for i in indices], label=category)

plt.title(f'Sentiment of Each Tweet - {COUNTRY}')
plt.xlabel('Tweet Index')
plt.ylabel('Sentiment Score')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(OUTPUT_DIR, f'{COUNTRY}_SentimentScatter.png'))

# Emotion Detection
emotions = detect_emotions(cleaned_tweets)

# Aggregate emotions
emotion_counts = defaultdict(int)
for emotion in emotions:
    emotion_label = emotion['label']
    emotion_counts[emotion_label] += 1

# Sort the emotions in descending order of counts
sorted_emotions = sorted(emotion_counts.items(), key=lambda item: item[1], reverse=True)
sorted_emotion_labels, sorted_emotion_values = zip(*sorted_emotions)

# Plotting Emotions with fixed axes
plt.figure(figsize=(10, 6))
plt.bar(sorted_emotion_labels, sorted_emotion_values)
plt.ylim(0, 4000)  # Setting the y-axis to scale up to 4000
plt.title('Distribution of Emotions in Tweets')
plt.xlabel('Emotion')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.savefig(os.path.join(OUTPUT_DIR, f'{COUNTRY}_Emotions.png'))

# Generate TF-IDF matrix
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(cleaned_tweets)

# Calculate Cosine Similarity Matrix
cosine_sim_matrix = cosine_similarity(tfidf_matrix)

# Perform Hierarchical Clustering
Z = linkage(cosine_sim_matrix, method='ward')

# Plotting the Dendrogram
plt.figure(figsize=(15, 10))
plt.title("Hierarchical Clustering Dendrogram")
plt.xlabel("Tweet Index")
plt.ylabel("Distance")
dendrogram(Z, leaf_rotation=90., leaf_font_size=8.)
plt.savefig(os.path.join(OUTPUT_DIR, f'{COUNTRY}_Dendrogram.png'))
# ------------------------------------ Evaluation ------------------------------------
tweets = read_json_files('Collected_Tweets/', 'Eval_Israel.json')
with open("Eval_Program.txt", 'a', encoding="UTF-8") as file:
    file.write(f"Average Sentiment Score for {COUNTRY}: {average_sentiment}")
    json.dump(average_sentiment, file, indent=4, sort_keys=True)
    file.write("\n")

for tweet in cleaned_tweets:
    score = sia.polarity_scores(tweet)['compound']
    with open("Eval_Program.txt", 'a', encoding="UTF-8") as file:
        json.dump(score, file, indent=4, sort_keys=True)
        file.write("\n")

def catagory_string(score):
    score = float(score)
    if score > 0:
        return 'positive'
    elif score < 0:
        return 'negative'
    else:
        return 'neutral'
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
# Load the TSV file
df = pd.read_csv('Collected_Tweets/Eval_Final.tsv', sep='\t', header=None)
df = df.iloc[:, :2]  # Select only the first two columns
df.columns = ['machine_label', 'human_label']
# Drop the first and last rows which contain averages
# df = df.drop([0, len(df)-1])

# Rename columns for clarity
df.columns = ['machine_label', 'human_label']

# Apply categorization
df['machine_label'] = df['machine_label'].apply(catagory_string)
df['human_label'] = df['human_label'].apply(catagory_string)

# Calculate confusion matrix
conf_matrix = confusion_matrix(df['human_label'], df['machine_label'], labels=['positive', 'neutral', 'negative'])

# Display confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=['positive', 'neutral', 'negative'])
disp.plot()
plt.savefig("Evaluation")

# For additional statistics like precision, recall
# You can use classification_report from sklearn.metrics
from sklearn.metrics import classification_report
# Prepare the text to be saved
output_text = ""
output_text += "Distribution of human labels:\n" + str(df['human_label'].value_counts()) + "\n\n"
output_text += "Distribution of machine labels:\n" + str(df['machine_label'].value_counts()) + "\n\n"

# Calculate classification report
report = classification_report(df['human_label'], df['machine_label'], 
                               target_names=['positive', 'neutral', 'negative'],
                               zero_division=0)
output_text += "Classification Report:\n" + report

# Save the output text to a file
with open("Classification_Report_and_Distribution.txt", "w") as file:
    file.write(output_text)
