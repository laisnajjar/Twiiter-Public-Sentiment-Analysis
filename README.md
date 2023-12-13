# Twitter Data Analysis Project

## Overview
This project consists of two main components: 
1. **Tweet Analysis (`tweet_analysis.py`):** Analyzes tweets for sentiment, emotion, and generates visualizations like word clouds and scatter plots.
2. **Tweet Search (`search_tweets.py`):** Searches and collects tweets based on specific search terms using the Twitter API.

The project aims to provide insights into public sentiment and opinions on topics related to Palestine and Israel, but can be customized for other topics.

## Features
- Sentiment analysis using NLTK.
- Emotion detection using a pre-trained DistilBERT model.
- Word cloud generation for visual representation of tweet content.
- Data visualization for sentiment and emotion analysis.
- Customizable Twitter data collection script.

## Requirements
This project requires Python and several Python packages. Install the required packages using:
pip install -r requirements.txt

The `requirements.txt` includes:
- numpy
- matplotlib
- wordcloud
- scikit-learn
- transformers
- nltk
- pandas
- requests (for `search_tweets.py`)

## Installation
1. Clone this repository to your local machine.
2. Navigate to the project directory.
3. Install the required packages:
pip install -r requirements.txt

## Configuration
### For Tweet Analysis
- Set the `COUNTRY` variable for the country-specific analysis.
- Adjust the `FILE_PATTERN`, `BASE_PATH`, `STOPWORDS_FILE`, and `OUTPUT_DIR` as needed.

### For Tweet Search
- Replace `search_term` with your desired search query.
- Ensure you have a Twitter Developer account and a Bearer Token.
- Store your Bearer Token in an environment variable `BEARER_TOKEN`.

## Usage
### Running Tweet Analysis
python tweet_analysis.py

Ensure your Bearer Token is correctly set in your environment variables.

## Output
- **Tweet Analysis:** Generates sentiment scatter plots, word clouds, emotion distribution charts, and more in the specified output directory.
- **Tweet Search:** Saves the collected tweets in JSON format in the `Collected_Tweets` directory.

## Contributing
Contributions to this project are welcome. Please fork the repository and submit a pull request with your changes.
