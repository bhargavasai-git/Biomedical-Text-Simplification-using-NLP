
from google.colab import drive
drive.mount('/content/drive')

import pandas as pd
train_df = pd.read_csv('/content/drive/My Drive/osfstorage PLABA_Dataset/train.csv')
test_df = pd.read_csv('/content/drive/My Drive/test (2) (1).csv')

train_df.tail()

test_df.tail()



train_df.info()
test_df.info()
print(train_df.isnull().sum())
print(test_df.isnull().sum())

import re

def clean_text(text):
  """Converts text to lowercase and removes non-alphanumeric characters."""
  if isinstance(text, str):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
  return text

train_df['input_text'] = train_df['input_text'].apply(clean_text)
train_df['target_text'] = train_df['target_text'].apply(clean_text)
test_df['input_text'] = test_df['input_text'].apply(clean_text)
test_df['target_text'] = test_df['target_text'].apply(clean_text)

display(train_df[['input_text', 'target_text']].head())
display(test_df[['input_text', 'target_text']].head())

import nltk
nltk.download('punkt_tab')
from nltk.tokenize import word_tokenize

train_df['input_tokens'] = train_df['input_text'].apply(word_tokenize)
train_df['target_tokens'] = train_df['target_text'].apply(word_tokenize)
test_df['input_tokens'] = test_df['input_text'].apply(word_tokenize)
test_df['target_tokens'] = test_df['target_text'].apply(word_tokenize)

display(train_df[['input_text', 'input_tokens', 'target_text', 'target_tokens']].head())
display(test_df[['input_text', 'input_tokens', 'target_text', 'target_tokens']].head())

import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def remove_stopwords(tokens):
  """Removes stopwords from a list of tokens."""
  return [word for word in tokens if word not in stop_words]

train_df['input_tokens'] = train_df['input_tokens'].apply(remove_stopwords)
train_df['target_tokens'] = train_df['target_tokens'].apply(remove_stopwords)
test_df['input_tokens'] = test_df['input_tokens'].apply(remove_stopwords)
test_df['target_tokens'] = test_df['target_tokens'].apply(remove_stopwords)

display(train_df[['input_tokens', 'target_tokens']].head())
display(test_df[['input_tokens', 'target_tokens']].head())

import nltk
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
nltk.download('omw-1.4')

lemmatizer = WordNetLemmatizer()

def lemmatize_tokens(tokens):
  """Lemmatizes a list of tokens."""
  return [lemmatizer.lemmatize(word) for word in tokens]

train_df['input_tokens'] = train_df['input_tokens'].apply(lemmatize_tokens)
train_df['target_tokens'] = train_df['target_tokens'].apply(lemmatize_tokens)
test_df['input_tokens'] = test_df['input_tokens'].apply(lemmatize_tokens)
test_df['target_tokens'] = test_df['target_tokens'].apply(lemmatize_tokens)

display(train_df[['input_tokens', 'target_tokens']].head())
display(test_df[['input_tokens', 'target_tokens']].head())

# Commented out IPython magic to ensure Python compatibility.
# %pip install transformers torch

train_df['combined_text'] = train_df['input_tokens'] + train_df['target_tokens']
train_df['combined_text'] = train_df['combined_text'].apply(lambda x: ' '.join(x))
display(train_df['combined_text'].head())

from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained('microsoft/biogpt')

# Commented out IPython magic to ensure Python compatibility.
# %pip install sacremoses

# Assuming 'tokenizer' object is already loaded from the previous step
# We need to tokenize the input and target texts for finetuning
from transformers import BioGptTokenizer

# Define a max_length based on typical sequence lengths for this model or task
# You might need to adjust this based on your data and computational resources
MAX_LENGTH = 512 # Example max length, adjust as needed

# Ensure tokenizer is loaded, or load it if not
if 'tokenizer' not in locals() or tokenizer is None:
    try:
        tokenizer = BioGptTokenizer.from_pretrained("microsoft/BioGPT")
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        # Handle the error, perhaps exit or skip tokenization
        raise e


def tokenize_function(examples):
    # Tokenize input text
    model_inputs = tokenizer(examples['input_text'], truncation=True, padding="max_length", max_length=MAX_LENGTH)

    # Tokenize target text
    labels = tokenizer(examples['target_text'], truncation=True, padding="max_length", max_length=MAX_LENGTH)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Convert the pandas DataFrame to a Hugging Face Dataset
from datasets import Dataset
train_dataset = Dataset.from_pandas(train_df)

# Apply the tokenization function to the dataset
tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)

print("Tokenized dataset example:")
print(tokenized_train_dataset[0])

from transformers import TrainingArguments, Trainer

# Define training arguments
# You can adjust these parameters based on your needs and computational resources
training_args = TrainingArguments(
    output_dir="./biogpt-finetuned",  # Output directory
    overwrite_output_dir=True,
    num_train_epochs=5,              # Number of training epochs
    per_device_train_batch_size=4,   # Batch size for training
    save_steps=10_000,               # Save model checkpoint every 10,000 steps
    save_total_limit=2,              # Only keep the latest 2 checkpoints
    logging_dir="./logs",            # Directory for storing logs
    logging_steps=500,
)

# Initialize the Trainer
trainer = Trainer(
    model=model,                         # The loaded BioGPT model
    args=training_args,                  # The training arguments
    train_dataset=tokenized_train_dataset, # The tokenized training dataset
)

# Start training
trainer.train()

import pandas as pd
from datasets import Dataset
import torch

# Ensure model and tokenizer are loaded and available
if 'model' not in locals() or 'tokenizer' not in locals():
    print("Error: Model or tokenizer not loaded. Please load them first.")
else:
    # Convert test_df to a Hugging Face Dataset
    test_dataset = Dataset.from_pandas(test_df)

    # Tokenize the test dataset (assuming tokenize_function is defined)
    tokenized_test_dataset = test_dataset.map(tokenize_function, batched=True)

    # Set the model to evaluation mode and move to GPU if available
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    generated_texts = [] # Initialize an empty list to store generated texts
    input_texts_list = [] # List to store original input texts
    target_texts_list = [] # List to store original target texts


    print("Generating text for the entire test dataset...")
    # Iterate through the entire tokenized test dataset and generate text
    for i, example in enumerate(tokenized_test_dataset):
        input_text = example['input_text']
        target_text = example['target_text']

        # Ensure input_ids is a tensor and has a batch dimension, move to device
        input_ids = torch.tensor(example['input_ids']).unsqueeze(0).to(device)

        # Generate text using the finetuned model
        with torch.no_grad():
            # Ensure attention_mask is also a tensor and has a batch dimension if used in generate, move to device
            attention_mask = torch.tensor(example['attention_mask']).unsqueeze(0).to(device)
            # Use max_new_tokens for generating text after the input, and add sampling
            output = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=100,  # Adjust as needed
                num_beams=5,
                early_stopping=True,
                do_sample=True,  # Enable sampling
                temperature=0.7 # Adjust temperature as needed
            )

        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

        # Append texts to lists
        input_texts_list.append(input_text)
        target_texts_list.append(target_text)
        generated_texts.append(generated_text)

        # Optional: Print progress or sample outputs
        if (i + 1) % 50 == 0 or i < 5:
            print(f"--- Generated Example {i+1}/{len(tokenized_test_dataset)} ---")
            print(f"Input Text: {input_text[:200]}...") # Print a snippet of input
            print(f"Generated Text: {generated_text[:200]}...") # Print a snippet of generated text
            print("-" * 20)

    print("\nFinished generating text.")

    # Create a new DataFrame with the collected texts
    results_df = pd.DataFrame({
        'input_text': input_texts_list,
        'target_text': target_texts_list,
        'generated_text': generated_texts
    })

    # Define the output CSV file path
    output_csv_path = '/content/drive/MyDrive/PLABA_outputs_alternative.csv' # Using a different name

    # Save the DataFrame to a CSV file
    results_df.to_csv(output_csv_path, index=False)

    print(f"Results saved to {output_csv_path}")
    display(results_df.head())

