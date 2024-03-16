import ctranslate2
import pandas as pd
from transformers import GPTNeoXForCausalLM, AutoTokenizer
import os
import shutil
import time

tokenizer = AutoTokenizer.from_pretrained(
  "google/flan-t5-small",
)


if os.path.exists("./model-out"):
    shutil.rmtree("./model-out")
    print(f"Previous model output folder has been deleted.")

converter = ctranslate2.converters.TransformersConverter("../flan-t5-small_final_model_no_shots")
converter.convert("./model-out")

translator = ctranslate2.Translator("./model-out", device="cpu", compute_type="auto")

while True:
    prompt = input("Enter your question (or 'q' to quit): ")
    
    if prompt.lower() == 'q':
        break
    
    start = time.time()
    prompt = "search: " + prompt

    tokens = tokenizer.convert_ids_to_tokens(tokenizer.encode(prompt))

    results = translator.translate_batch([tokens], 
                                         beam_size=1,
                                         max_decoding_length=25,
                                         sampling_topk=50,
                                         sampling_topp=0.1,
                                         sampling_temperature=0.7,
                                         repetition_penalty=1.1,
                                         use_vmap=True)

    output_tokens = results[0].hypotheses[0]
    output_text = tokenizer.decode(tokenizer.convert_tokens_to_ids(output_tokens), skip_special_tokens=True)

    end = time.time()

    print(f"Time taken (ms): {(end - start) * 1000}")
    print("total tokens:", len(output_tokens))
    print(output_text)
    print(output_text.split("; "))