import argparse
import sys
import pandas as pd
import os
from peft import PeftModel
from transformers import LlamaForCausalLM, LlamaTokenizer, AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM 
from tqdm import tqdm
from numpy import argmax
import torch
from sklearn.metrics import accuracy_score
from util_prompt import prepare_data


TOKEN = 'your_huggingface_token'

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--load_8bit", action='store_true')
    parser.add_argument("--share_gradio", action='store_true')
    parser.add_argument("--base_model", type=str, help="Path to pretrained model", required=True)
    parser.add_argument("--lora_weights", type=str, default="x")
    parser.add_argument("--lang_alpa", type=str, default="ar")
    parser.add_argument("--lang_prompt", type=str, default="ar")
    parser.add_argument("--output_folder", type=str, default="output")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    os.makedirs(args.output_folder, exist_ok=True)
    tokenizer_class = LlamaTokenizer if 'llama' in args.base_model else AutoTokenizer
    model_class = LlamaForCausalLM if 'llama' in args.base_model else AutoModelForCausalLM

    SAVE_FILE = f'{args.output_folder}/result_prompt_{args.lang_prompt}_alpa_{args.lang_alpa}_{args.base_model.split("/")[-1]}.csv'
    tokenizer = tokenizer_class.from_pretrained(args.base_model, use_auth_token=TOKEN)
    
    if 'mt0' in args.base_model or 'arat5' in args.base_model.lower():
        model = AutoModelForSeq2SeqLM.from_pretrained(args.base_model, device_map="auto", load_in_8bit="xxl" in args.base_model)
        from util_compute import predict_classification_mt0_by_letter as predict_classification
    else:
        model = model_class.from_pretrained(args.base_model, load_in_8bit=args.load_8bit, trust_remote_code=True, device_map="auto", use_auth_token=TOKEN)
        from util_compute import predict_classification_causal_by_letter as predict_classification
    
    # Load adapter if we use adapter
    if args.lora_weights != "x":
        model = PeftModel.from_pretrained(
            model,
            args.lora_weights,
            torch_dtype=torch.float16,
        )
        SAVE_FILE = f'{args.output_folder}/result_prompt_{args.lang_prompt}_alpa_{args.lang_alpa}_{args.lora_weight.split("/")[-1]}.csv'

    # unwind broken decapoda-research config
    if 'llama' in args.base_model:
        model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
        model.config.bos_token_id = 1
        model.config.eos_token_id = 2

    model.eval()
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)
    
    inputs, golds, outputs_options = prepare_data(args)
    preds = []
    probs = []
    for idx in tqdm(range(len(inputs))):
        conf, pred = predict_classification(model, tokenizer, inputs[idx], outputs_options[idx], device, args.lang_alpa)
        probs.append(conf)
        preds.append(pred)

    output_df = pd.DataFrame()
    output_df['input'] = inputs
    output_df['golds'] = golds
    output_df['options'] = outputs_options
    output_df['preds'] = preds
    output_df['probs'] = probs
    output_df.to_csv(SAVE_FILE, index=False)

if __name__ == "__main__":
    main()
