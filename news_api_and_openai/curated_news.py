import os
import openai
from newsapi import NewsApiClient

openai.api_key = os.getenv("OPENAI_API_KEY")
newsapi_key = os.getenv("NEWSAPI_KEY")

def get_completion(prompt):
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0,
    )
    return response.choices[0].message["content"]

newsapi = NewsApiClient(api_key=newsapi_key)
top_headlines = newsapi.get_top_headlines(sources='espn', language='en')

prompt = f"""
Remove the URL
Create an HTML file with heading and content using CSS table-layout
Do not paraphrase and add the entire content
Add alternating background colors to each row: light grey and dark grey
Show only the title and the description fields
Add a header at the top of the page with the source and date
Keep the content within an A4 size sheet
Review:```{top_headlines}```
"""

response = get_completion(prompt)
with open("news.html", "w") as f:
    f.write(response)
