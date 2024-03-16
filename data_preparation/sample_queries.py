import json
import pandas as pd
import openai
from tqdm import tqdm
import re
import os
import random
from pydantic import BaseModel
from typing import List
from fireworks.client import Fireworks
from dotenv import load_dotenv

load_dotenv()


client = Fireworks(api_key=os.getenv("FIREWORKS_API_KEY"))

# Define JSON schema for model response
class QueryItem(BaseModel):
    inner_monologue: str
    queries: List[str]


class QueryList(BaseModel):
    items: List[QueryItem]



with open("few_shot_queries.txt", "r") as f:
    FEW_SHOTS_EXAMPLES = f.read()


def build_prompt_chat(good_examples, batched_requests):
    # requests_str = "\n".join([f"- {request}" for request in batched_requests])
    requests_str = f'{{"items": {json.dumps([{"request": request} for request in batched_requests])}}}'
    prompt = f"""### Instructions
For each user request, generate a set of relevant web search queries that would help the user finding information to address the request by typing these queries into a web search engine. For simple, straightforward requests, generate around 1-2 queries. For more complex or multi-part requests, generate around 3-4 queries. I have attached some good examples in the "Few-shot examples" section for your reference. The search queries should be concise, focused (rather than complete sentences), and cover different aspects of the user request to maximize the chances of finding relevant information. Also, avoid giving away the complete answer in the queries. Think step-by-step and first explain your reasoning and the intent behind each query in the "inner_monologue" field of the JSON object. Then output the final set of queries in the "queries" field as a JSON list.

Please provide your response in the following JSON format:
{{
  "items": [
    {{
      "inner_monologue": "Your step-by-step reasoning and thought process behind the queries",
      "queries": [
        "Query 1",
        "Query 2",
        "..."
      ]
    }},
    ...
  ]
}}

### Few-shot examples
{good_examples}

### User requests
{requests_str}

### Search queries
"""
    return prompt


# init good examples
good_examples = FEW_SHOTS_EXAMPLES
parsed_few_shots = json.loads(FEW_SHOTS_EXAMPLES)
prompt_set = set()

df = pd.read_csv('cleaned_requests.csv')
batch_size = 10
data = []
MAX_RETRY = 4

def generate_queries_batch(batched_requests, retry_count=0):
    try:
        model_completion = client.chat.completions.create(
            model="accounts/fireworks/models/mixtral-8x7b-instruct",
            response_format={"type": "json_object", "schema": QueryList.model_json_schema()},
            max_tokens=2048,
            temperature=0.75,
            top_p=1.0,
            top_k=50,
            messages=[
                {"role": "system", "content": f"You are a web search expert with great Google-Fu that always output a single JSON object called 'items'."},
                {"role": "user", "content": build_prompt_chat(good_examples, batched_requests),}
            ],
        )

        response = model_completion.choices[0].message.content
        response = response.replace('\_', '_')
        # print(response)

    
        parsed_response = QueryList.model_validate_json(response).items
       
        if len(parsed_response) != len(batched_requests):
            raise ValueError("Mismatched number of search query items and user requests.")
    
        return parsed_response
    except Exception as e:
        print(f"Error generating search queries for batch: {str(e)}")
        if retry_count < MAX_RETRY:
            print(f"Retrying batch (attempt {retry_count + 1})...")
            return generate_queries_batch(batched_requests, retry_count + 1)
        else:
            print("Max retries exceeded. Filling in dummy data.")
            dummy_data = [{"inner_monologue": "dummy_monologue", "search_query": ["quries"]} for _ in batched_requests]
            return dummy_data


for i in tqdm(range(0, len(df), batch_size)):
    batched_requests = df["request"].iloc[i:i+batch_size].tolist()
    parsed_response = generate_queries_batch(batched_requests)
    data.extend(parsed_response)
 
df = pd.concat([df, pd.DataFrame.from_records(data)], axis=1)

df.to_csv('mixtral_requests_and_queries.csv', index=False)
