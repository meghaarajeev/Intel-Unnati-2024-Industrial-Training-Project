from flask import Flask, request, render_template
import os
import joblib
import torch
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification
import PyPDF2
import re
import pandas as pd
import spacy
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Loading the joblib model
model = joblib.load('model.joblib')

# Loading the tokenizer and BERT model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertForSequenceClassification.from_pretrained('bert-base-uncased')

# Load SpaCy model for NER
nlp = spacy.load("en_core_web_sm")

def extract_text_from_pdf(pdf_path):
    text = ''
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            text += page.extract_text()
    return text

def structure_text(raw_text):
    clauses = re.split(r'\n\d+\.\s', raw_text)
    structured_text = {f"Clause {i+1}": clause for i, clause in enumerate(clauses)}
    return structured_text

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('index.html', error='No file part')
    file = request.files['file']
    if file.filename == '':
        return render_template('index.html', error='No selected file')
    if file and allowed_file(file.filename):
        filename = file.filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # Extract text from the PDF
            raw_text = extract_text_from_pdf(filepath)
            
            # Structure the extracted text
            structured_text = structure_text(raw_text)
            
            # Convert structured text to a DataFrame
            data = {'text': [], 'label': []}
            for clause, text in structured_text.items():
                data['text'].append(text)
                data['label'].append(clause)
            df = pd.DataFrame(data)

            # Preprocess the user contract text
            user_contract_text = list(structured_text.values())
            user_contract_encodings = tokenizer(user_contract_text, truncation=True, padding=True, max_length=128, return_tensors="pt")

            # Perform prediction
            bert_model.eval()
            with torch.no_grad():
                outputs = bert_model(**user_contract_encodings)
                predictions = outputs.logits
                pred_labels = torch.argmax(predictions, dim=1).numpy()

            def perform_ner(text):
                doc = nlp(text)
                entities = [(ent.text, ent.label_) for ent in doc.ents]
                return entities
            
            def compare_text_with_template(text, template_file):
                templates = pd.read_csv(template_file)
                comparison_results = []
                for index, row in templates.iterrows():
                    section = row['section']
                    template_text = str(row['template_text'])
                    match = "Match" if template_text.strip() in text.strip() else "No Match"
                    comparison_results.append((section, match))
                return comparison_results

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

            # Perform Named Entity Recognition on the user contract
            user_contract_ner_results = perform_ner(raw_text)

            # Compare the user contract with the template
            user_contract_comparison_results = compare_text_with_template(raw_text, 'template.csv')

            # Summarize the comparison results
            user_contract_summary = summarize_comparison_results(user_contract_comparison_results)
            all_clauses_valid = all(label == "Match" for _, label in user_contract_comparison_results)

            summary = {
                "Total Clauses": user_contract_summary['total_sections'],
                "Valid Clauses": user_contract_summary['match_count'],
                "Invalid Clauses": user_contract_summary['no_match_count'],
                "Validity": 'Valid' if all_clauses_valid else 'Invalid',
                "Similarity Percentage": round((user_contract_summary['match_count'] / user_contract_summary['total_sections']) * 100, 2)
            }
            return render_template('result.html', result=summary)
        except Exception as e:
            return render_template('index.html', error=str(e))
    else:
        return render_template('index.html', error='File type not allowed')

if __name__ == '__main__':
    app.run(debug=True)

    