import pandas as pd
import torch
from tqdm import tqdm

def load_model(model_path):
    # Load the trained model from the specified path
    model = torch.load(model_path)
    model.eval()
    return model

def load_tokenizer(tokenizer_path):
    # Load the tokenizer from the specified path
    tokenizer = torch.load(tokenizer_path)
    return tokenizer

def make_predictions(model, tokenizer, input_data, max_length=512):
    # Prepare the input data for the model
    inputs = tokenizer(input_data, return_tensors="pt", truncation=True, max_length=max_length).to(model.device)
    
    # Generate predictions
    with torch.no_grad():
        output_ids = model.generate(inputs['input_ids'], max_new_tokens=32, do_sample=False)
    
    # Decode the predictions
    predictions = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return predictions

def run_inference(test_data_path, model_path, tokenizer_path):
    # Load the test data
    test_df = pd.read_csv(test_data_path)
    
    # Load the model and tokenizer
    model = load_model(model_path)
    tokenizer = load_tokenizer(tokenizer_path)
    
    # Create a list to store predictions
    all_predictions = []
    
    # Iterate over the test data and make predictions
    for question in tqdm(test_df['question'], desc="Running inference"):
        prediction = make_predictions(model, tokenizer, question)
        all_predictions.append(prediction)
    
    # Add predictions to the DataFrame
    test_df['predictions'] = all_predictions
    
    return test_df

if __name__ == "__main__":
    # Define paths
    test_data_path = '../data/test_f.csv'
    model_path = 'path/to/your/trained/model.pth'
    tokenizer_path = 'path/to/your/tokenizer.pth'
    
    # Run inference
    results = run_inference(test_data_path, model_path, tokenizer_path)
    print(results)