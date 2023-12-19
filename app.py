# app.py
import os
from flask import Flask, request, jsonify
from dotenv import load_dotenv
import pandas as pd
import openai
from langchain.vectorstores import FAISS

app = Flask(__name__)
load_dotenv()

# # Initialize OpenAI API
# class GenerativeModel:
#     def __init__(self, model_type, api_key=None):
#         self.model_type = model_type
#         self.api_key = api_key

#         if model_type == "gpt3.5-turbo" and not api_key:
#             raise ValueError("OPENAI_API_KEY is required for GPT-3.5-turbo")

#     def generate_text(self, prompt):
#         try:
#             if self.model_type == "gpt3.5-turbo":
#                 response = self.model.Completion.create(
#                     prompt=prompt,
#                     max_tokens=150,
#                     n=1,
#                     stop=None,
#                     temperature=0.7,
#                 )
#                 return response.choices[0].text.strip()
#             elif self.model_type == "gpt2":
#                 input_ids = tokenizer.encode(prompt, return_tensors="pt")
#                 output = model.generate(input_ids, max_length=100)
#                 generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
#                 return generated_text
#             elif self.model_type == "llama2":
#                 # Add code for llama2 model inference here
#                 input_ids = tokenizer.encode(prompt, return_tensors="pt")
#                 output = model.generate(input_ids, max_length=100)
#                 generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
#                 return generated_text
#             # Add conditions for other models as needed
#         except OpenAIAPIError as e:
#             return f"Error: {str(e)}"

# Initialize the model
MODEL_TYPE = os.getenv("MODEL_TYPE")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def get_completion(text,prompt, model="gpt-3.5-turbo"):
    text = text[:4090] + "..." if len(text) > 4090 else text

    prompt = f""" Please follow the instruction given in  {prompt} and give your response below for following give context:
    ```{text}```
    """
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0, # this is the degree of randomness of the model's output
    )
    return response.choices[0].message["content"]



# # Flask routes
# @app.route("/generate", methods=["POST"])
# def generate():
#     data = request.get_json()
#     prompt = data.get("prompt", "")

#     if not prompt:
#         return jsonify({"error": "Prompt is required"}), 400

#     generated_text = model.generate_text(prompt)
#     return jsonify({"generated_text": generated_text})

@app.route("/api/v1/llm", methods=["POST"])
def inference():
    data = request.get_json()
    embeddings = data.get("embeddings")
    prompt = data.get("prompt")
    model = data.get("modelName")

    if  not prompt or not model:
        return jsonify({"error": "Invalid request. Missing inputData, prompt, or model."}), 400

    try:
        inputData = FAISS.load_local("faiss_index", embeddings)

        # docs = new_db.similarity_search(query)
        if model == "gpt3.5-turbo":
            response = get_completion(inputData, prompt, model="gpt-3.5-turbo")
            return jsonify({"response": response})

        elif model == "llama2":
            response = get_completion(inputData, prompt, model="llama2")
            return jsonify({"response": response})

        elif model == "mistral":
            response = get_completion(inputData, prompt, model="mistral")
            return jsonify({"response": response})

        else:
            return jsonify({"error": "Invalid model. Supported models are gpt3.5-turbo, llama2, and mistral."}), 400

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host= 'localhost', port = 5000 , debug=True)
