import ctranslate2
import gradio as gr
from transformers import AutoTokenizer
import os
import shutil
import time

tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")

if os.path.exists("./model-out"):
    shutil.rmtree("./model-out")
    print("Previous model output folder has been deleted.")

converter = ctranslate2.converters.TransformersConverter("./fine-tuned-flan-t5-small")
converter.convert("./model-out")

translator = ctranslate2.Translator("./model-out", device="cpu", compute_type="int8_float32")

def generate_queries(prompt):
    start = time.time()
    prompt = prompt.strip()
    prompt = "search: " + prompt
    input_tokens = tokenizer.convert_ids_to_tokens(tokenizer.encode(prompt))[:45]
    results = translator.translate_batch([input_tokens],
                                         #max_input_length=50,
                                         beam_size=1,
                                         max_decoding_length=30,
                                         sampling_topk=40,
                                         sampling_topp=0.25,
                                         sampling_temperature=0.7,
                                         repetition_penalty=1.15,
                                         use_vmap=True)
    output_tokens = results[0].hypotheses[0]
    output_text = tokenizer.decode(tokenizer.convert_tokens_to_ids(output_tokens), skip_special_tokens=True)
    
    deduped_output = list(set(output_text.split("; ")))

    end = time.time()
    time_taken = round((end - start) * 1000, 2)
    total_output_tokens = len(output_tokens)
    total_input_tokens = len(input_tokens)

    # Convert the answer to a bullet point list
    bullet_points = "- " + "\n- ".join(deduped_output)
    truncated_input = tokenizer.decode(tokenizer.convert_tokens_to_ids(input_tokens), skip_special_tokens=True)
    return bullet_points, time_taken, total_output_tokens, total_input_tokens, truncated_input

interface = gr.Interface(
    fn=generate_queries,
    inputs=gr.Textbox(lines=2, placeholder="Enter your question", label="Question"),
    outputs=[
        gr.Textbox(label="Queries"),
        gr.Textbox(label="Time Taken (ms)"),
        gr.Textbox(label="Total Output Tokens"),
        gr.Textbox(label="Total Input Tokens"),
        gr.Textbox(label="Actual Model Input")
    ],
    title="Query Reformulation",
    description="Enter a question and get a number of search engine queries.",
    allow_flagging="never"
)

interface.launch(auth=("user", os.getenv("GRADIO_PASSWORD")))