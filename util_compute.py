import torch
import numpy as np

alpa_ar = {
    0: 'أ',
    1: 'ب',
    2: 'ج',
    3: 'د',
    4: 'ه'
}
alpa_en = {
    0: 'A',
    1: 'B',
    2: 'C',
    3: 'D',
    4: 'E'
}


def softmax(x):
    z = x - max(x)
    numerator = np.exp(z)
    denominator = np.sum(numerator)
    softmax = numerator/denominator
    return softmax


def predict_classification_causal_by_letter(model, tokenizer, input_text, labels, device, lang_alpa):
    alpa = alpa_ar
    if lang_alpa == 'en':
        alpa = alpa_en

    choices = list(alpa.values())[:len(labels)]
    choice_ids = [tokenizer.encode(choice)[-1] for choice in choices]
    with torch.no_grad():
        
        if model.config._name_or_path in ['core42/jais-30b-v3', 'core42/jais-30b-chat-v3']:
            inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=2048)
        elif model.config._name_or_path in ['aubmindlab/aragpt2-mega']:
            inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=1024)
        else:
            inputs = tokenizer(input_text, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        if model.config._name_or_path in ['FreedomIntelligence/AceGPT-13B', 'FreedomIntelligence/AceGPT-7B', 'FreedomIntelligence/AceGPT-7B-chat', 'FreedomIntelligence/AceGPT-13B-chat']:
            inputs.pop("token_type_ids")
        outputs = model(**inputs, labels=input_ids)
        last_token_logits = outputs.logits[:, -1, :]
        choice_logits = last_token_logits[:, choice_ids].detach().cpu().numpy()
        conf = softmax(choice_logits[0])
        pred = alpa[np.argmax(choice_logits[0])]
    return conf, pred


def predict_classification_mt0_by_letter(model, tokenizer, input_text, labels, device, lang_alpa):
    alpa = alpa_ar
    if lang_alpa == 'en':
        alpa = alpa_en

    choices = list(alpa.values())[:len(labels)]
    choice_ids = [tokenizer.encode(choice)[0] for choice in choices]
    with torch.no_grad():
        start_token = tokenizer('<pad>', return_tensors="pt").to(device)
        inputs = tokenizer(input_text, return_tensors="pt").to(device)
        outputs = model(**inputs, decoder_input_ids=start_token['input_ids'])
        last_token_logits = outputs.logits[:, -1, :]
        choice_logits = last_token_logits[:, choice_ids].detach().cpu().numpy()
        conf = softmax(choice_logits[0])
        pred = alpa[np.argmax(choice_logits[0])]
    return conf, pred

