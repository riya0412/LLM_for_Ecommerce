# Install the transformers library
# !pip install transformers

import torch
from transformers import BertForMaskedLM, BertTokenizer, BertConfig

# Define your pre-training text corpus
corpus=["I need a new pair of running shoes","I'm looking for a new laptop for work","I want to buy a new smartphone","I'm looking for a new TV for my living room","I need a new refrigerator","I want to buy a new vacuum cleaner","I want to buy a new smart speaker","a new pair of sunglasses","I want to buy a new pair of jeans","I'm looking for a new pair of hiking boots","I want to buy a new basketbal","looking for a new tennis racke"]

# Initialize the tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForMaskedLM(config=BertConfig())

# Tokenize and convert text to tensor
input_ids = tokenizer(corpus, return_tensors="pt", padding=True, truncation=True)["input_ids"]

# Masking some of the tokens
masking_prob = 0.15
masked_indices = torch.bernoulli(torch.full(input_ids.shape, masking_prob)).bool()
input_ids[masked_indices] = tokenizer.mask_token_id

# Define the batch size and number of epochs for training
batch_size = 32
num_epochs = 3

# Define optimizer and loss function (example, you might need to fine-tune)
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
loss_fn = torch.nn.CrossEntropyLoss()

# Training loop
for epoch in range(num_epochs):
    for i in range(0, len(input_ids), batch_size):
        optimizer.zero_grad()
        batch_input_ids = input_ids[i:i+batch_size]
        outputs = model(batch_input_ids)
        logits = outputs.logits
        loss = loss_fn(logits.view(-1, logits.shape[-1]), batch_input_ids.view(-1))
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')

# Save the pre-trained model
model.save_pretrained('bert_model')
tokenizer.save_pretrained('bert_model_tokernizer')
