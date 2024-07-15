

import os
import PyPDF2
import re

def extract_text_from_pdf(pdf_path):
    text = ''
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text()
    return text

def structure_text(text):
    clauses = re.split(r'\n\d+\.\s', text)
    structured_text = {f"Clause {i+1}": clause.strip() for i, clause in enumerate(clauses)}
    return structured_text

def parse_contracts(directory_path):
    contracts = {}
    for root, _, files in os.walk(directory_path):
        for file in files:
            if file.endswith('.pdf'):
                pdf_path = os.path.join(root, file)
                raw_text = extract_text_from_pdf(pdf_path)
                structured_text = structure_text(raw_text)
                contracts[pdf_path] = structured_text
    return contracts

directory_path = '/content/Business-Contract-Dataset-Intel-Training--Program-2024'
contracts = parse_contracts(directory_path)

for contract_path, structured_text in contracts.items():
    print(f"\nContract: {contract_path}")
    for clause, text in structured_text.items():
        print(f"{clause}: {text}")

# Optional: Save the structured text to a JSON file for further use
import json
output_path = 'structured_contracts.json'
with open(output_path, 'w') as f:
    json.dump(contracts, f, indent=4)

pip install transformers[torch] accelerate -U

from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import torch
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import classification_report

# Load data
data = {
    "text": [text for contract in contracts.values() for text in contract.values()],
    "label": [label for contract in contracts.values() for label in contract.keys()]
}

df = pd.DataFrame(data)
label_map = {label: i for i, label in enumerate(df['label'].unique())}
df['label'] = df['label'].map(label_map)

# Split data
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)

# Tokenize text
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
train_encodings = tokenizer(list(X_train), truncation=True, padding=True, max_length=128)
test_encodings = tokenizer(list(X_test), truncation=True, padding=True, max_length=128)

class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = Dataset(train_encodings, y_train.tolist())
test_dataset = Dataset(test_encodings, y_test.tolist())

# Load model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(label_map))

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset
)

# Train model
trainer.train()

# Evaluate model
results = trainer.evaluate()
print(results)

predictions, _, _ = trainer.predict(test_dataset)
pred_labels = np.argmax(predictions, axis=1)

print(classification_report(y_test, pred_labels, labels=list(label_map.values()), target_names=label_map.keys()))

!pip install PyMuPDF

import fitz  # PyMuPDF

def highlight_deviations(pdf_path, deviations):
    doc = fitz.open(pdf_path)
    for page_num, deviation in deviations.items():
        page = doc.load_page(page_num)
        for inst in deviation:
            rect = page.search_for(inst['text'])
            for r in rect:
                highlight = page.add_highlight_annot(r)
                highlight.update()
    output_path = pdf_path.replace(".pdf", "_highlighted.pdf")
    doc.save(output_path)
    doc.close()
    return output_path

# Example usage
deviations = {
    0: [{'text': 'example deviation', 'label': 'Clause 2'}]
}
highlighted_pdf_path = highlight_deviations("/content/Business-Contract-Dataset-Intel-Training--Program-2024/partnership/partnership_3.pdf", deviations)

pip install pymupdf pytesseract pillow easyocr enchant

import fitz  # PyMuPDF
import easyocr

def ocr_pdf(pdf_path):
    text = ''
    try:
        # Initialize EasyOCR reader
        reader = easyocr.Reader(['en'])  # Specify languages as needed

        # Open the PDF file
        pdf_document = fitz.open(pdf_path)

        # Iterate through each page
        for page_num in range(len(pdf_document)):
            page = pdf_document.load_page(page_num)

            # Convert PDF page to an image
            pix = page.get_pixmap()
            img = pix.tobytes("png")

            # Perform OCR on the image
            result = reader.readtext(img, detail=0)

            # Combine results into a single string
            text += ' '.join(result) + '\n'

        # Close the PDF document
        pdf_document.close()

    except Exception as e:
        print(f"Error processing PDF: {e}")

    return text

# Example usage:
pdf_path = "/content/Business-Contract-Dataset-Intel-Training--Program-2024/partnership/partnership_3.pdf"
ocr_text = ocr_pdf(pdf_path)
print(ocr_text)

!pip install spacy

#NER
import spacy

def perform_ner(text):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return entities

# Example usage
ner_results = perform_ner(ocr_text)
print(ner_results)

import pandas as pd

def compare_text_with_template(text, template_file):
    # Load the template CSV file
    templates = pd.read_csv(template_file)

    # Initialize an empty list to store comparison results
    comparison_results = []

    # Iterate through each row in the template
    for index, row in templates.iterrows():
        section = row['section']
        template_text = str(row['template_text'])  # Ensure template_text is treated as a string

        # Check if the template text is found in the OCR text
        if template_text.strip() in text.strip():
            match = "Match"
        else:
            match = "No Match"

        # Append the result to the list
        comparison_results.append((section, match))

    return comparison_results

# Example usage with placeholder OCR text and template file
# ocr_text = """
# This is a sample OCR text. It should contain various clauses and sections that might match with the template.
# The OCR text could have different formatting or slight variations compared to the template text.
# """

template_file = 'template.csv'
comparison_results = compare_text_with_template(ocr_text, template_file)

# Print comparison results for each section
for section, result in comparison_results:
    print(f"{section}: {result}")

# prompt: using all the model training and functions created so far print the whole summary of whether the business contract given by the user valid or not

import pandas as pd
import numpy as np
# Load the PDF file from the user
pdf_path = "/content/Business-Contract-Dataset-Intel-Training--Program-2024/partnership/partnership_3.pdf"

# Extract text from the PDF
raw_text = extract_text_from_pdf(pdf_path)

# Structure the extracted text
structured_text = structure_text(raw_text)

# Convert structured text to a DataFrame
data = {'text': [], 'label': []}
for contract_path, structured_text in contracts.items():
    for clause, text in structured_text.items():
        data['text'].append(text)
        data['label'].append(clause)

df = pd.DataFrame(data)

# Preprocess the user contract text
user_contract_text = list(structured_text.values())
user_contract_encodings = tokenizer(user_contract_text, truncation=True, padding=True, max_length=128)
user_contract_dataset = Dataset(user_contract_encodings, [0] * len(user_contract_text))

# Predict the labels for the user contract
predictions, _, _ = trainer.predict(user_contract_dataset)
pred_labels = np.argmax(predictions, axis=1)

# Perform Named Entity Recognition on the user contract
user_contract_ner_results = perform_ner(raw_text)

# Compare the user contract with the template
user_contract_comparison_results = compare_text_with_template(raw_text, template_file)

# Summarize the comparison results
user_contract_summary = summarize_comparison_results(user_contract_comparison_results)

# Check if all clauses in the user contract are valid
all_clauses_valid = all(label == "Match" for _, label in user_contract_comparison_results)

# Print the summary
print("\nSummary:")
print(f"Total Clauses: {user_contract_summary['total_sections']}")
print(f"Valid Clauses: {user_contract_summary['match_count']}")
print(f"Invalid Clauses: {user_contract_summary['no_match_count']}")
print(f"Validity: {'Valid' if all_clauses_valid else 'Invalid'}")

import joblib
joblib.dump(model, 'modelfinal.joblib')

# prompt: definition of summarize_comparison_results function also return validity

def summarize_comparison_results(comparison_results):
  total_sections = len(comparison_results)
  match_count = sum(result == "Match" for _, result in comparison_results)
  no_match_count = total_sections - match_count
  validity = "Valid" if match_count == total_sections else "Invalid"
  return {
    "total_sections": total_sections,
    "match_count": match_count,
    "no_match_count": no_match_count,
    "validity": validity
  }
