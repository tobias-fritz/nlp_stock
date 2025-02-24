from newsapi import NewsApiClient
import datetime

def get_last_month_stock_sentiment(ticker: str, 
                                   classifier: callable, 
                                   api_key: str
                                ) -> list[dict, str, str]:
    '''Get the sentiment of the news articles for the last month for a given stock.'''

    today = datetime.datetime.now()
    date_range = {
        'from_param': (today - datetime.timedelta(days=30)).strftime("%Y-%m-%d"),
        'to': today.strftime("%Y-%m-%d")
    }
    
    api = NewsApiClient(api_key=api_key)
    articles = [article for page in range(1,6) 
               for article in api.get_everything(
                   q=ticker, language='en', sort_by='relevancy', 
                   page=page, **date_range)['articles']]
    
    return [result for article in articles 
            if (result := _clean_article(article, classifier)) is not None]

def _clean_article(article: str, classifier: callable) -> dict:
    '''Clean the article and return the date and sentiment if the sentiment quality is high.'''
    try:
        # Get the sentiment of the article
        sentiment = classifier(article["description"])[0]
        # If the sentiment score is high, return the date and sentiment
        if sentiment["score"] > 0.9:
            return {
                "date": datetime.datetime.strptime(
                    article["publishedAt"], "%Y-%m-%dT%H:%M:%SZ"
                ).strftime("%Y-%m-%d"),
                "sentiment": sentiment["label"]
            }
    except:
        return None
