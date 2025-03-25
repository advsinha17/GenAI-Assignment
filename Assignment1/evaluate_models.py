import torch
from datasets import load_dataset
import evaluate
from tqdm import tqdm

def calculate_scores(model, en_tokenizer, hi_tokenizer, max_length=100):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    
    dataset = load_dataset('cfilt/iitb-english-hindi')['test']
    
    bleu = evaluate.load('bleu')
    rouge = evaluate.load('rouge')
    meteor = evaluate.load('meteor')
    
    predictions = []
    references = []
    
    for example in tqdm(dataset, desc="Evaluating"):
        en_encoding = en_tokenizer(
            example['translation']['en'],
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        ).to(device)
        
        with torch.no_grad():
            output = model.predict(en_encoding['input_ids'], en_encoding['attention_mask'], max_length=max_length)

        prediction = hi_tokenizer.decode(output[0], skip_special_tokens=True)
        reference = example['translation']['hi']
        
        predictions.append(prediction)
        references.append(reference)
    
    bleu_results = bleu.compute(
        predictions=predictions, 
        references=[[ref] for ref in references]
    )
    
    rouge_results = rouge.compute(
        predictions=predictions, 
        references=references
    )
    
    meteor_results = meteor.compute(
        predictions=predictions, 
        references=references
    )
    
    return {
        'bleu': bleu_results['bleu'],
        'rouge': rouge_results['rougeL'],
        'meteor': meteor_results['meteor'],
        'samples': [
            {'input': dataset[i]['translation']['en'],
             'prediction': predictions[i],
             'reference': references[i]}
            for i in range(min(3, len(predictions)))
        ]
    }