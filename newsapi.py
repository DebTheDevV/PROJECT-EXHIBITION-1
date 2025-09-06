import requests
response = requests.get("https://newsapi.org/v2/top-headlines?country=us&apiKey=your_newsapi_key")
print(response.json())