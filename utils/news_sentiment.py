import os
from newsapi import NewsApiClient
from textblob import TextBlob

def get_news_sentiment(company_name):
    """Fetch news and calculate sentiment for a given company."""
    
    # Initialize NewsAPI client with default key if environment variable not set
    api_key = os.getenv('NEWS_API_KEY', 'your_default_key')
    newsapi = NewsApiClient(api_key=api_key)
    
    try:
        # Get news articles
        articles = newsapi.get_everything(
            q=company_name,
            language='en',
            sort_by='publishedAt',
            page_size=5
        )
        
        news_data = []
        for article in articles['articles']:
            # Calculate sentiment using TextBlob
            sentiment = TextBlob(article['description'] if article['description'] else '').sentiment.polarity
            
            # Convert sentiment score to label
            if sentiment > 0.1:
                sentiment_label = "Positive 📈"
            elif sentiment < -0.1:
                sentiment_label = "Negative 📉"
            else:
                sentiment_label = "Neutral 📊"
            
            news_data.append({
                'title': article['title'],
                'description': article['description'],
                'source': article['source']['name'],
                'sentiment': sentiment_label
            })
            
        return news_data
    
    except Exception as e:
        return [{
            'title': 'Error fetching news',
            'description': str(e),
            'source': 'System',
            'sentiment': 'Neutral 📊'
        }]
