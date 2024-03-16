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
class RequestItem(BaseModel):
    request: str

class RequestList(BaseModel):
    items: List[RequestItem]


if os.path.exists("repeated_requests.log"):
    os.remove("repeated_requests.log")
    print(f"repeated requests log has been deleted.")

def log_to_file(message, filename="repeated_requests.log"):
    """Append a message to a specified file.
    
    Args:
        message (str): The message to log.
        filename (str): The file to which the message should be appended.
    """
    with open(filename, 'a') as file:
        file.write(message + '\n')

model_name = "mistralai/Mixtral-8x7B-Instruct-v0.1"

NUM_SAMPLES = 10000


# FEW_SHOTS_EXAMPLES = 
with open("few_shot_requests.txt", "r") as f:
    FEW_SHOTS_EXAMPLES = f.read()


def build_prompt_chat(good_examples):
    prompt = f"""### Instructions
Generate 10 user requests that are common for web search engine users. Each request should be between 10 to 30 words long. Try to be as diverse as possible in topics (e.g., history, science, technology, sports, entertainment, fashion, finance, travel, art, food, etc.) and question types, including factual questions, comparisons, open-ended queries, and requests for information presentation (e.g., tables, lists). Do not restrict yourself to requests on recent events. The generated requests should be standalone and not refer to the examples provided. Please output the generated requests in valid JSON format. Here are some good examples of the desired styles:

### Examples
{good_examples}
"""
    return prompt


# init good examples
good_examples = FEW_SHOTS_EXAMPLES
parsed_few_shots = json.loads(FEW_SHOTS_EXAMPLES)
prompt_set = set()


def generate_few_shots(templates, values):
    few_shots = []
    for template, value_set in zip(templates, values):
        value = random.choice(value_set)
        request = template.format(*value) if isinstance(value, tuple) else template.format(value)
        few_shots.append({"request": request})
    return few_shots

# Example usage
templates = [
    "In what year was the winner of the {}th edition of the {} competition born?",
    "Who lived longer, {} or {}?",
    "Author {} has collaborated with a {} who served as the {} under which President?",
    "Create a table for top {} that are {}",
    "What are some ways to do fast {}?",
    "How many {} are there in {}?",
    "What is the capital city of {}?",
    "Explain the concept of {} in simple terms.",
    "What are the main differences between {} and {}?",
    "Suggest a {} recipe that includes {}."
]

values = [
    [("44", "Miss World"), ("23", "Miss Universe"), ("55", "Academy Awards")],
    [("Nikola Tesla", "Milutin Milankovic"), ("Albert Einstein", "Stephen Hawking"), ("Thomas Edison", "Alexander Graham Bell")],
    [("David Chanoff", "U.S. Navy admiral", "ambassador to the United Kingdom"), ("Michael Lewis", "former U.S. President", "Secretary of State"), ("Malcolm Gladwell", "renowned scientist", "advisor to the White House")],
    [("noise cancelling headphones", "affordable"), ("gaming laptops", "lightweight"), ("smartwatches", "durable")],
    ["query reformulation", "image compression", "text summarization"],
    [("provinces", "Canada"), ("states", "Australia"), ("counties", "Ireland")],
    ["Brazil", "Japan", "Egypt"],
    ["quantum computing", "blockchain", "artificial intelligence"],
    [("Python", "JavaScript"), ("socialism", "capitalism"), ("Impressionism", "Expressionism")],
    [("vegan", "tofu"), ("Italian", "pasta"), ("Mexican", "avocado")]
]



data = []


for i in tqdm(range(NUM_SAMPLES // 10)):
    model_completion = client.chat.completions.create(
        model="accounts/fireworks/models/mixtral-8x7b-instruct",
        response_format={"type": "json_object", "schema": RequestList.model_json_schema()},
        max_tokens=1024,
        temperature=0.8,
        top_p=1.0,
        top_k=50,
        messages=[
            {"role": "system", "content": f"You are a helpful assistant that is expert in web search, designed to output a single JSON object called 'items'. The current year is {random.randint(2000, 2024)}."},
            {"role": "user", "content": build_prompt_chat(good_examples),}
        ],
    )
    
    
    response = model_completion.choices[0].message.content
    response = response.replace('\_', '_')

    try:
        parsed_response = json.loads(response)["items"]
    except Exception as e:
        print(response)
        print(e)
        continue

    data.extend(parsed_response)

    sampled_dicts = random.sample(data, min(len(data), 10)) + generate_few_shots(templates, values)
    random.shuffle(sampled_dicts)
    good_examples = json.dumps(sampled_dicts)


    for item in parsed_response:
        if item["request"] in prompt_set:
            log_to_file(f"Prompt repetition: {item['request']}")
        prompt_set.add(item["request"])
 


df = pd.DataFrame(data, columns=["request"])
df.to_csv('requests.csv', index=False)