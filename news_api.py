import requests


url = "https://newsapi.org/v2/everything?q=&from=2026-03-10&to=2026-03-10&sortBy=similarity?country=us&category=business&apiKey=API_KEY"

response = requests.get(url)