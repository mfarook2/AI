import pandas as pd
import os
from pandasai import PandasAI
from pandasai.llm.openai import OpenAI

df = pd.read_csv('smartwatches.csv')

#print(df.to_string())

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
llm = OpenAI(api_token=OPENAI_API_KEY)
pandas_ai = PandasAI(llm)

print(pandas_ai.run(df, prompt='Which is the most expensive Brand'))
print(pandas_ai.run(df, prompt='Which brand has the best quality based on customer ratings and battery life only'))
print(pandas_ai.run(df, prompt='List the brands in terms of quality, where best quality is listed first'))
print(pandas_ai.run(df, prompt='build a line chart based for each brand for quality, make is visually appealing'))
print(pandas_ai.run(df, prompt='build a pie chart based for each brand for quality, make is visually appealing'))
