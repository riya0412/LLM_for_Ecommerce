from transformers import BertTokenizer, BertForTokenClassification
import torch
import pandas as pd

# Load pre-trained BERT model for NER
model = BertForTokenClassification.from_pretrained("bert_model")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def search_products(query):
    # Tokenize and convert to tensor
    inputs = tokenizer(query, return_tensors="pt")

    # Perform inference
    with torch.no_grad():
        outputs = model(**inputs)

    # Get predicted labels
    predictions = torch.argmax(outputs.logits, dim=2)
    predicted_labels = [tokenizer.convert_ids_to_tokens(pred) for pred in predictions[0].tolist()]

    # Extract product names
    # df=pd.read_csv("product.csv")
    products = []
    current_product = ""
    for token, label in zip(tokenizer.tokenize(query), predicted_labels):
        if label.startswith("B-"):
            if current_product:
                products.append(current_product)
            current_product = token
        elif label.startswith("I-"):
            current_product += " " + token
        else:
            if current_product:
                products.append(current_product)
            current_product = ""

    if current_product:
        products.append(current_product)

    return products

query = input()
results = search_products(query)

if results:
    print("Matching products found:")
    for product in results:
        print(product)
else:
    print("No matching products found.")
