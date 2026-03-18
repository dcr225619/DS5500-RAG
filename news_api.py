import requests
from news_key import news_key
from newsapi import NewsApiClient

def call_news_api(q, start_date, end_date, qintitle=None, sources=None, domains=None, exclude_domains=None, language='en', sortBy='relevancy'):
    """
    call news API to get data

    Args:
    CATEGORIES = {"business", "entertainment", "general", "health", "science", "sports", "technology"}
    SORT_METHOD = {"relevancy", "popularity", "publishedAt"}

    Returns:
    Json
    """
    newsapi = NewsApiClient(api_key=news_key)

    # # /v2/top-headlines
    # top_headlines = newsapi.get_top_headlines(q='bitcoin',
    #                                         sources='bbc-news,the-verge',
    #                                         category='business',
    #                                         language='en',
    #                                         country='us')

    # /v2/everything
    all_articles = newsapi.get_everything(q=q,
                                        qintitle=qintitle,
                                        sources=sources,
                                        domains=domains,
                                        exclude_domains=exclude_domains,
                                        from_param=start_date,
                                        to=end_date,
                                        language=language,
                                        sort_by=sortBy,
                                        page=2)

    return all_articles

if __name__ == '__main__':
    temp = call_news_api(q='bitcoin',
                        start_date='2026-03-10',
                        end_date='2026-03-17',
                        language='en',
                        sortBy='relevancy')

    print(temp)