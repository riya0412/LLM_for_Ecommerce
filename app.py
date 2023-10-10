from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import pandas as pd

# app = Flask(__name__)

# Load pre-trained model and tokenizer
model_name = "textattack/bert-base-uncased-imdb"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
search_pipeline = pipeline("text-classification", tokenizer=tokenizer, model=model)

# @app.route('/search', methods=['POST'])
def search(query):
    # query = request.get_json('query')
    df=pd.read_csv("product.csv")
    product_titles = list(df["product_name"])

    encoded_inputs = tokenizer.batch_encode_plus(
        product_titles,
        truncation=True,
        padding=True,
        return_tensors="pt"
    )
    encoded_query = tokenizer.encode_plus(
        query,
        truncation=True,
        padding=True,
        return_tensors="pt"
    )
    input_ids = encoded_inputs["input_ids"]
    attention_mask = encoded_inputs["attention_mask"]
    query_input_ids = encoded_query["input_ids"]
    query_attention_mask = encoded_query["attention_mask"]

    outputs = model(
        query_input_ids,
        attention_mask=query_attention_mask,
        input_ids=input_ids,
        attention_mask2=attention_mask
    )

    logits = outputs["logits"]
    similarity_scores = logits[:, 1].tolist()  # Column 1 represents the similarity scores
    top_matches_indices = sorted(range(len(similarity_scores)), key=lambda i: similarity_scores[i], reverse=True)[:10]
    top_matches = [product_titles[i] for i in top_matches_indices]

    return jsonify({'top_matches': top_matches})

query=input()
result=search(query)
print(result)

# if __name__ == '__main__':
#     app.run(debug=True)