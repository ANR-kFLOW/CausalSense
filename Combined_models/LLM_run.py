import logging
import os
import re
import pandas as pd
import yaml
import openai
import time
from argparse import ArgumentParser
from transformers import AutoModelForCausalLM, AutoTokenizer

from langchain_core.prompts.prompt import PromptTemplate

# === Init ====================================================================

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Specify the GPU id
os.environ["HUGGINGFACE_HUB_CACHE"] = "/data/huggingface/"

available_llms = {
    "zephyr": "HuggingFaceH4/zephyr-7b-beta",
    "dpo": "yunconglong/Truthful_DPO_TomGrc_FusionNet_7Bx2_MoE_13B",
    "una": "fblgit/UNA-TheBeagle-7b-v1",
    "solar": "bhavinjawade/SOLAR-10B-OrcaDPO-Jawade",
    "gpt4": "OpenAI-GPT4"  # Added GPT-4
}

loggingFormatString = (
    "%(asctime)s:%(levelname)s:%(threadName)s:%(funcName)s:%(message)s"
)

logging.basicConfig(format=loggingFormatString, level=logging.INFO)


def load_prompt_template(template_path):
    with open(template_path, 'r') as file:
        template_data = yaml.safe_load(file)
    return PromptTemplate(
        input_variables=template_data['input'], template=template_data['template']
    )


def call_gpt4_api(prompt, api_key, max_retries=3, timeout=1200):
    openai.api_key = api_key
    retries = 0
    while retries < max_retries:
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",  # Use GPT-4 model
                #model = 'gpt-3.5-turbo',
                messages=[
                    {"role": "system",
                     "content": "You are an assistant that extracts subjects, objects, and relations from sentences."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=150,  # Maximum number of tokens in the output
                request_timeout=timeout  # Increase the timeout limit
            )
            return response.choices[0].message['content'].strip()
        except openai.error.Timeout as e:
            retries += 1
            print(f"Request timed out. Retrying {retries}/{max_retries}...")
            time.sleep(2 ** retries)  # Exponential backoff
        except Exception as e:
            print(f"An error occurred: {e}")
            break
    return None


def call_huggingface_model(prompt, tokenizer, model):


    inputs = tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True)
    outputs = model.generate(**inputs, max_length=150)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return response


def extract_information(input_sentence, examples, llm_model, template_path, api_key, verbose,model_name):
    PROMPT_TEMPLATE = load_prompt_template(template_path)
    input_data = {"input_sentence": input_sentence, "examples": examples}
    formatted_prompt = PROMPT_TEMPLATE.format(**input_data)

    if llm_model == "gpt4":
        res = call_gpt4_api(formatted_prompt, api_key)
    else:
        tokenizer = AutoTokenizer.from_pretrained( available_llms[model_name])
        model = AutoModelForCausalLM.from_pretrained( available_llms[model_name])
        res = call_huggingface_model(formatted_prompt,tokenizer,model)
    print(res)
    patterns = [r"Subject:\s*(.*?),\s*Object:\s*(.*?),\s*Relation:\s*(.*)"]
    match = re.search(patterns[0], res)

    if match:
        subject, object_, relation = match.groups()
        return {"subject": subject, "object": object_, "relation": relation}
    else:
        return {}


def get_few_shot_examples(news_dataset_path, num_examples):
    news_dataset = pd.read_csv(news_dataset_path)
    examples = []
    relations = ['cause', 'enable', 'intend', 'prevent']

    for relation in relations:
        relation_examples = news_dataset[news_dataset['relation'] == relation].sample(num_examples)
        for _, row in relation_examples.iterrows():
            example_text = f"Sentence: \"{row['text']}\" -> Subject: {row['subject']}, Object: {row['object']}, Relation: {row['relation']}"
            examples.append(example_text)

    return "\n".join(examples)


def run(task, news_dataset_path, test_dataset_path, num_examples, llm_model, template_path, output_path, api_key=None,
        verbose=False, df=None):
    if task == 'test':
        examples = get_few_shot_examples(news_dataset_path, num_examples)
        if df.empty:
            test_dataset = pd.read_csv(test_dataset_path)
        else:
            print('true')
            test_dataset = df
        if os.path.exists(output_path):
            try:
                results = pd.read_csv(output_path).to_dict('records')
            except pd.errors.EmptyDataError:
                results = []
        else:
            results = []
        df_end = pd.DataFrame()
        data_frames = []
        for _, row in test_dataset.iterrows():
            input_sentence = row['text']

            extracted_data = extract_information(input_sentence, examples, llm_model, template_path, api_key, verbose,llm_model)
            result = {
                "text": input_sentence,
                "subject": extracted_data.get("subject", "N/A"),
                "object": extracted_data.get("object", "N/A"),
                "relation": extracted_data.get("relation", "N/A"),
            }
            results.append({
                "text": input_sentence,
                "subject": extracted_data.get("subject", "N/A"),
                "object": extracted_data.get("object", "N/A"),
                "relation": extracted_data.get("relation", "N/A"),
            })
            data_frames.append(pd.DataFrame([result]))
            #df.append(pd.DataFrame(results))
            # Save results incrementally
            pd.DataFrame(results).to_csv(output_path, index=False)
    df_end = pd.concat(data_frames, ignore_index=True)
    return True, df_end
def run_LLM(args):
    
    logging.basicConfig(format=loggingFormatString, level=args.LLMS_log)

    run_response, df = run(
        args.LLMS_task,
        args.LLMS_news_dataset,
        args.LLMS_test_dataset,
        args.LLMS_num_examples,
        args.LLMS_llm,
        args.LLMS_template,
        args.LLMS_output,
        args.LLMS_api_key,
        args.LLMS_verbose,
        args.only_causal
    )
    df.to_csv('semifinaltest.csv')
    return df
    

if __name__ == '__main__':
    parser = ArgumentParser(
        prog='LLM4ke',
        description='Extract subject, object, and relation from sentences using Hugging Face models or GPT-4'
    )
    parser.add_argument('--task', help='Task to perform', choices=['test'])
    parser.add_argument('--news_dataset', help='Path to the news dataset CSV file', required=True)
    parser.add_argument('--test_dataset', help='Path to the test dataset CSV file', required=True)
    parser.add_argument('--num_examples', type=int, help='Number of examples per relation', required=True)
    parser.add_argument('--llm', help='LLM to use', default='zephyr', choices=available_llms)
    parser.add_argument('--template', help='Path to the prompt template YAML file', required=True)
    parser.add_argument('--output', help='Path to save the output predictions CSV file', required=True)
    parser.add_argument('--api_key', help='API key for GPT-4', required=False)
    parser.add_argument('--verbose', help='Print the full prompt', default=False, action='store_true')
    parser.add_argument("--log", type=int, choices=[10, 20, 30, 40, 50], action="store", default=20,
                        help="Verbosity (default: INFO) : DEBUG = 10, INFO = 20, WARNING = 30, ERROR = 40, CRITICAL = 50")

    args = parser.parse_args()
    logging.basicConfig(format=loggingFormatString, level=args.LLMS_log)

    run_response, df = run(
        args.LLMS_task,
        args.LLMS_news_dataset,
        args.LLMS_test_dataset,
        args.LLMS_num_examples,
        args.LLMS_llm,
        args.LLMS_template,
        args.LLMS_output,
        args.LLMS_api_key,
        args.LLMS_verbose
    )
    df.to_csv('semifinaltest.csv')
