

import pandas as pd
train_df = pd.read_csv('/content/train.csv')
test_df = pd.read_csv('/content/test (2).csv')

display(train_df.head())
display(test_df.head())

import nltk
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.corpus import wordnet

# Download necessary NLTK data
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')
try:
    nltk.data.find('corpora/omw-1.4')
except LookupError:
    nltk.download('omw-1.4')
try:
    nltk.data.find('stemmers/porter')
except LookupError:
    nltk.download('porter_stemmer')

lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

def lemmatize_text(text):
    if isinstance(text, str):
        return ' '.join([lemmatizer.lemmatize(word) for word in text.split()])
    return text

def stem_text(text):
    if isinstance(text, str):
        return ' '.join([stemmer.stem(word) for word in text.split()])
    return text

train_df['lemmatized_input_text'] = train_df['input_text'].apply(lemmatize_text)
train_df['lemmatized_target_text'] = train_df['target_text'].apply(lemmatize_text)
test_df['lemmatized_input_text'] = test_df['input_text'].apply(lemmatize_text)
test_df['lemmatized_target_text'] = test_df['target_text'].apply(lemmatize_text)

train_df['stemmed_input_text'] = train_df['input_text'].apply(stem_text)
train_df['stemmed_target_text'] = train_df['target_text'].apply(stem_text)
test_df['stemmed_input_text'] = test_df['input_text'].apply(stem_text)
test_df['stemmed_target_text'] = test_df['target_text'].apply(stem_text)

display(train_df.head())
display(test_df.head())

import spacy

# Load a spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading en_core_web_sm model. This may take a few minutes...")
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

def analyze_text_spacy(text):
    if not isinstance(text, str):
        return [], []
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    dependencies = [(token.text, token.dep_, token.head.text) for token in doc]
    return entities, dependencies

# Apply the function to the lemmatized text columns
train_df['ner_results'], train_df['dependency_results'] = zip(*train_df['lemmatized_input_text'].apply(analyze_text_spacy))
test_df['ner_results'], test_df['dependency_results'] = zip(*test_df['lemmatized_input_text'].apply(analyze_text_spacy))

display(train_df.head())
display(test_df.head())

!pip install sacremoses

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BioGptTokenizer, BioGptForCausalLM

# Instantiate tokenizer and model
tokenizer = BioGptTokenizer.from_pretrained("microsoft/BioGPT")
model = BioGptForCausalLM.from_pretrained("microsoft/BioGPT")

# Tokenize the data
max_length = 512  # Define a suitable max length

def tokenize_function(examples):
    # Tokenize input text
    tokenized_input = tokenizer(
        examples['lemmatized_input_text'],
        padding="max_length",
        truncation=True,
        max_length=max_length
    )
    # Tokenize target text
    tokenized_target = tokenizer(
        examples['lemmatized_target_text'],
        padding="max_length",
        truncation=True,
        max_length=max_length
    )

    # Prepare labels: shift input tokens to the right
    # This is required for causal language modeling where the model predicts the next token
    labels = tokenized_target["input_ids"].copy()
    # Replace padding token id in labels with -100 so it's ignored in loss computation
    labels = [[(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels]

    return {
        'input_ids': tokenized_input['input_ids'],
        'attention_mask': tokenized_input['attention_mask'],
        'labels': labels
    }

# Convert dataframes to dictionaries for easier processing
train_data_dict = train_df.to_dict(orient='list')
test_data_dict = test_df.to_dict(orient='list')


# Tokenize the dataframes
train_tokenized = tokenize_function(train_data_dict)
test_tokenized = tokenize_function(test_data_dict)

# Create PyTorch Datasets
class TextDataset(Dataset):
    def __init__(self, tokenized_data):
        self.tokenized_data = tokenized_data

    def __len__(self):
        return len(self.tokenized_data['input_ids'])

    def __getitem__(self, idx):
        return {
            'input_ids': torch.tensor(self.tokenized_data['input_ids'][idx]),
            'attention_mask': torch.tensor(self.tokenized_data['attention_mask'][idx]),
            'labels': torch.tensor(self.tokenized_data['labels'][idx])
        }

train_dataset = TextDataset(train_tokenized)
test_dataset = TextDataset(test_tokenized)

# Optionally, create DataLoaders (useful for batching during training)
# train_dataloader = DataLoader(train_dataset, batch_size=8)
# test_dataloader = DataLoader(test_dataset, batch_size=8)

print("Train dataset size:", len(train_dataset))
print("Test dataset size:", len(test_dataset))

from transformers import TrainingArguments, Trainer

# Fast training arguments (for testing or quick runs)
training_args = TrainingArguments(
    output_dir="./biogpt_fast_run",
    overwrite_output_dir=True,
    num_train_epochs=1,                 # Only 1 epoch
    per_device_train_batch_size=2,      # Small batch size
    save_strategy="no",                 # No checkpoints          # Disable evaluation
    logging_strategy="no",              # Disable logging
    report_to="none",                   # No external logging
    fp16=False,                         # Use True if supported GPU
    dataloader_num_workers=0,           # Fewer workers = faster startup
    max_steps=10,                       # Limit steps (super fast)
)

# Subset of dataset for speed
train_dataset = train_dataset

# Instantiate Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)

print("Training started (fast mode)...")
trainer.train()
print("Training finished.")

# Optionally save the model (can skip to save time)
trainer.save_model("./biogpt_fast_model")

# Generate text using the fine-tuned model on the test set
model.eval() # Set the model to evaluation mode

generated_texts = []
# num_rows_to_generate = 148 # Number of rows to generate text for

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

with torch.no_grad(): # Disable gradient calculation for inference
    # for i in range(min(num_rows_to_generate, len(test_dataset))):
    for i in range(len(test_dataset)):
        input_data = test_dataset[i]
        input_ids = input_data['input_ids'].unsqueeze(0).to(device) # Add batch dimension and move to device
        attention_mask = input_data['attention_mask'].unsqueeze(0).to(device) # Add batch dimension and move to device

        # Generate text
        output = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=128,  # Set the maximum number of new tokens to generate
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            early_stopping=True
        )

        # Decode the generated text
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        generated_texts.append(generated_text)

# Display the original and generated text for the first 30 rows
for i in range(min(len(generated_texts), len(test_df))): # Display up to the number of generated texts or test_df length
    print(f"Original Input Text (Row {i+1}):\n{test_df['input_text'][i]}")
    print(f"Generated Text (Row {i+1}):\n{generated_texts[i]}\n{'-'*50}\n")

from google.colab import drive
drive.mount('/content/drive')

# Create a DataFrame with original and generated texts
generated_df = pd.DataFrame({
    'original_input_text': test_df['input_text'][:num_rows_to_generate],
    'generated_text': generated_texts
})

# Define the path to save the CSV in Google Drive
output_path = '/content/drive/MyDrive/generated_texts.csv'

# Save the DataFrame to CSV
generated_df.to_csv(output_path, index=False)

print(f"Generated texts saved to {output_path}")

!pip install sari
!pip install textstat
!pip install rouge_score
!pip install bert_score

import textstat
from rouge_score import rouge_scorer
from bert_score import score as bert_score

def calculate_fkgl(text):
    """Calculates the Flesch-Kincaid Grade Level of a text."""
    if not isinstance(text, str) or len(text.split()) < 10: # textstat might fail on very short texts
        return 0.0
    try:
        return textstat.flesch_kincaid_grade(text)
    except Exception: # Catch potential errors from textstat
        return 0.0

def calculate_rouge_scores(reference_text, generated_text):
    """Calculates ROUGE scores between a reference and generated text."""
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference_text, generated_text)
    return {
        'rouge1': scores['rouge1'].fmeasure,
        'rouge2': scores['rouge2'].fmeasure,
        'rougeL': scores['rougeL'].fmeasure
    }

def calculate_bert_scores(generated_texts, reference_texts):
    """Calculates BERT scores between lists of generated and reference texts."""
    # bert_score.score expects lists of strings
    P, R, F1 = bert_score(generated_texts, reference_texts, lang='en', verbose=False)
    return {'precision': P.mean().item(), 'recall': R.mean().item(), 'f1': F1.mean().item()}

def calculate_sari(original_text, generated_text, target_text):
    """
    Calculates the SARI score between original, generated, and target texts.
    This is a simplified implementation as the 'sari' library could not be installed.
    """
    def tokenize(text):
        return text.lower().split()

    original_tokens = tokenize(original_text)
    generated_tokens = tokenize(generated_text)
    target_tokens = tokenize(target_text)

    # Identify added, deleted, and kept words
    added_tokens = [token for token in generated_tokens if token not in original_tokens]
    deleted_tokens = [token for token in original_tokens if token not in generated_tokens]
    kept_tokens_in_generated = [token for token in generated_tokens if token in original_tokens]
    kept_tokens_in_original = [token for token in original_tokens if token in generated_tokens]

    # Calculate precision, recall, and F1 for added words
    # Precision for added: proportion of added words in generated text that are in target text
    added_precision_numerator = sum(1 for token in added_tokens if token in target_tokens)
    added_precision = added_precision_numerator / len(added_tokens) if added_tokens else 0

    # Recall for added: proportion of words in target text not in original text that are in added words
    words_in_target_not_in_original = [token for token in target_tokens if token not in original_tokens]
    added_recall_numerator = sum(1 for token in added_tokens if token in words_in_target_not_in_original)
    added_recall = added_recall_numerator / len(words_in_target_not_in_original) if words_in_target_not_in_original else 0

    added_f1 = 2 * (added_precision * added_recall) / (added_precision + added_recall) if (added_precision + added_recall) > 0 else 0

    # Calculate precision, recall, and F1 for deleted words
    # Precision for deleted: proportion of deleted words that are NOT in target text
    deleted_precision_numerator = sum(1 for token in deleted_tokens if token not in target_tokens)
    deleted_precision = deleted_precision_numerator / len(deleted_tokens) if deleted_tokens else 0

    # Recall for deleted: proportion of words in original text not in target text that are in deleted words
    words_in_original_not_in_target = [token for token in original_tokens if token not in target_tokens]
    deleted_recall_numerator = sum(1 for token in deleted_tokens if token in words_in_original_not_in_target)
    deleted_recall = deleted_recall_numerator / len(words_in_original_not_in_target) if words_in_original_not_in_target else 0

    deleted_f1 = 2 * (deleted_precision * deleted_recall) / (deleted_precision + deleted_recall) if (deleted_precision + deleted_recall) > 0 else 0

    # Calculate precision, recall, and F1 for kept words
    # Precision for kept: proportion of kept words in generated text that are in target text
    kept_precision_numerator = sum(1 for token in kept_tokens_in_generated if token in target_tokens)
    kept_precision = kept_precision_numerator / len(kept_tokens_in_generated) if kept_tokens_in_generated else 0

    # Recall for kept: proportion of words in original text that are in target text and also in kept words
    words_in_original_and_target = [token for token in original_tokens if token in target_tokens]
    kept_recall_numerator = sum(1 for token in kept_tokens_in_original if token in words_in_original_and_target)
    kept_recall = kept_recall_numerator / len(words_in_original_and_target) if words_in_original_and_target else 0


    kept_f1 = 2 * (kept_precision * kept_recall) / (kept_precision + kept_recall) if (kept_precision + kept_recall) > 0 else 0

    # SARI is the average of F1 for added, deleted, and kept
    sari_score = (added_f1 + deleted_f1 + kept_f1) / 3

    return sari_score

print("Evaluation functions defined.")

# Initialize new columns in the generated_df DataFrame to store the calculated scores
generated_df['fkgl_score'] = 0.0
generated_df['rouge1_score'] = 0.0
generated_df['rouge2_score'] = 0.0
generated_df['rougel_score'] = 0.0
generated_df['sari_score'] = 0.0

# Iterate through each row of the generated_df DataFrame and calculate scores
for index, row in generated_df.iterrows():
    original_input_text = row['original_input_text']
    generated_text = row['generated_text']
    # We need the target_text from the test_df for evaluation metrics
    target_text = test_df.loc[index, 'target_text']

    # Calculate FKGL score
    generated_df.loc[index, 'fkgl_score'] = calculate_fkgl(generated_text)

    # Calculate ROUGE scores
    rouge_scores = calculate_rouge_scores(target_text, generated_text)
    generated_df.loc[index, 'rouge1_score'] = rouge_scores['rouge1']
    generated_df.loc[index, 'rouge2_score'] = rouge_scores['rouge2']
    generated_df.loc[index, 'rougel_score'] = rouge_scores['rougeL']

    # Calculate SARI score
    generated_df.loc[index, 'sari_score'] = calculate_sari(original_input_text, generated_text, target_text)

# Calculate BERT scores for the entire set of generated and target texts
bert_scores = calculate_bert_scores(generated_df['generated_text'].tolist(), test_df['target_text'][:num_rows_to_generate].tolist())

# Store the overall BERT scores (these are averages)
overall_bert_precision = bert_scores['precision']
overall_bert_recall = bert_scores['recall']
overall_bert_f1 = bert_scores['f1']

print("Evaluation scores calculated and added to the generated_df DataFrame.")
print(f"Overall BERT Precision: {overall_bert_precision:.4f}")
print(f"Overall BERT Recall: {overall_bert_recall:.4f}")
print(f"Overall BERT F1: {overall_bert_f1:.4f}")

display(generated_df.head())

# Calculate the mean scores for each metric
average_fkgl = generated_df['fkgl_score'].mean()
average_rouge1 = generated_df['rouge1_score'].mean()
average_rouge2 = generated_df['rouge2_score'].mean()
average_rougel = generated_df['rougel_score'].mean()
average_sari = generated_df['sari_score'].mean()

# Display the average scores
print("Average Evaluation Scores:")
print(f"  FKGL: {average_fkgl:.4f}")
print(f"  ROUGE:")
print(f"    Rouge-1: {average_rouge1:.4f}")
print(f"    Rouge-2: {average_rouge2:.4f}")
print(f"    Rouge-L: {average_rougel:.4f}")
print(f"  SARI: {average_sari:.4f}")
print(f"  BERT Score (Overall):")
print(f"    Precision: {overall_bert_precision:.4f}")
print(f"    Recall: {overall_bert_recall:.4f}")
print(f"    F1: {overall_bert_f1:.4f}")
