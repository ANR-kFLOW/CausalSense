import io
import time
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, GenerationConfig, PretrainedConfig
import torch
from torch.utils.data import Dataset, DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cuda_flag = False
if torch.cuda.is_available():
  cuda_flag = True
else:
  cuda_flag = False
class DataSequence(Dataset):
  def __init__(self, data):
    self.data = data
  def __len__(self):
    return len(self.data)
  def __getitem__(self, idx):
    return self.data[idx]
def test_model(df, path_to_model):
  # Load model
  with open(path_to_model, 'rb') as f:
    buffer = io.BytesIO(f.read())
  if cuda_flag == True:
    model = torch.load(buffer).to(device)
  else:
    model = torch.load(buffer,map_location=torch.device('cpu')).to(device)
  
  model_config = PretrainedConfig.from_pretrained('rebel_config/', cache_dir='data/huggingface/')
  config = GenerationConfig.from_model_config(model_config)
  print(config)
  
  # Load tokenizer
  tokenizer = AutoTokenizer.from_pretrained('Babelscape/rebel-large', cache_dir='data/huggingface/')
  # Load data
  #data = pd.read_csv(data_csv)
  data = df
  #sentences = data['sentence'].tolist()
  sentences = data['text'].tolist()
  # Prepare dataset
  test_dataset = DataSequence(sentences)
  test_dataloader = DataLoader(test_dataset, batch_size=4)
  model.eval()
  # Start timing the inference process
  start_time = time.time()
  pred = []
  for sentence in test_dataloader:
    inputs = tokenizer(sentence, return_tensors='pt', padding=True, truncation=True, max_length=512).to(device)
    # Perform inference
    with torch.no_grad():
      #outputs = model.generate(**inputs)
      outputs = model.generate(generation_config=config,**inputs)
    outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    print(outputs)
    pred.extend(outputs)
  # End timing the inference process
  end_time = time.time()
  total_time = end_time - start_time
  print(f"Total inference time: {total_time} seconds")
  # Save predictions to a CSV file
  results_df = pd.DataFrame({'sentence': sentences, 'prediction': pred})
  results_df.to_csv('rebel_prediction/pred_rebel.csv', index=False)
  return results_df