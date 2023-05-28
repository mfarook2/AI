import pandas as pd
from pandasai import PandasAI
from pandasai.llm.openai import OpenAI

df = pd.read_csv('smartwatches.csv')

#print(df.to_string())

OPENAI_API_KEY = "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"
llm = OpenAI(api_token=OPENAI_API_KEY)
pandas_ai = PandasAI(llm)

#print(pandas_ai.run(df, prompt='Which is the most expensive Brand'))
print(pandas_ai.run(df, prompt='Which brand has the best quality based on customer ratings battery life only'))
#print(pandas_ai.run(df, prompt='List the brands in terms of quality, where best quality is listed first'))