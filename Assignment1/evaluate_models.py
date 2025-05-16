import torch
import numpy as np
import evaluate
import matplotlib.pyplot as plt
from datasets import load_dataset
from tqdm import tqdm

def calculate_scores(model, en_tokenizer, hi_tokenizer, max_length=100):
    """
    Evaluate the model on the IITB English-Hindi dataset and calculate BLEU, ROUGE, and METEOR scores.
    Args:
        model: The trained model.
        en_tokenizer: The English tokenizer.
        hi_tokenizer: The Hindi tokenizer.
        max_length: The maximum length for tokenization.
    Returns:
        A dcitionary containing the BLEU, ROUGE, and METEOR scores, as well as the average scores by input sequence length and some sample translations.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    
    dataset = load_dataset('cfilt/iitb-english-hindi')['test']

    bleu = evaluate.load('bleu')
    rouge = evaluate.load('rouge')
    meteor = evaluate.load('meteor')

    predictions, references = [], []
    sequence_lengths, bleu_scores, rouge_scores, meteor_scores = [], [], [], []

    for example in tqdm(dataset, desc="Evaluating"):
        input_text = example['translation']['en']
        input_length = len(input_text.split())
        
        en_encoding = en_tokenizer(
            input_text, max_length=max_length, padding='max_length', truncation=True, return_tensors='pt'
        ).to(device)
        
        with torch.no_grad():
            output = model.predict(en_encoding['input_ids'], en_encoding['attention_mask'], max_length=max_length)

        prediction = hi_tokenizer.decode(output[0], skip_special_tokens=True)
        reference = example['translation']['hi']

        bleu_result = bleu.compute(predictions=[prediction], references=[[reference]])
        rouge_result = rouge.compute(predictions=[prediction], references=[reference])
        meteor_result = meteor.compute(predictions=[prediction], references=[reference])

        predictions.append(prediction)
        references.append(reference)
        sequence_lengths.append(input_length)
        bleu_scores.append(bleu_result['bleu'])
        rouge_scores.append(rouge_result['rougeL'])
        meteor_scores.append(meteor_result['meteor'])

    # Binning by sequence length
    bins = [0, 10, 20, 30, 40, 50, 75]  
    bin_indices = np.digitize(sequence_lengths, bins)

    def compute_avg_scores(scores):
        bin_avgs, bin_counts = [], []
        for bin_idx in range(1, len(bins)):
            mask = (bin_indices == bin_idx)
            if sum(mask) > 0:
                avg_score = np.mean([scores[i] for i in range(len(scores)) if mask[i]])
                bin_avgs.append(avg_score)
                bin_counts.append(sum(mask))
            else:
                bin_avgs.append(0)
                bin_counts.append(0)
        return bin_avgs, bin_counts

    bleu_avgs, counts = compute_avg_scores(bleu_scores)
    rouge_avgs, _ = compute_avg_scores(rouge_scores)
    meteor_avgs, _ = compute_avg_scores(meteor_scores)

    def plot_score_distribution(score_avgs, ylabel, title, filename):
        plt.figure(figsize=(12, 6))
        plt.bar(bins[:-1], score_avgs, width=8, align='edge')
        plt.xlabel('Sequence Length')
        plt.ylabel(ylabel)
        plt.title(title)
        plt.xticks(bins[:-1], [f'{bins[i]}-{bins[i+1]}' for i in range(len(bins)-1)], rotation=45)

        plt.tight_layout()
        plt.savefig(f'/content/drive/MyDrive/{filename}.png')
        plt.show()


    plot_score_distribution(bleu_avgs, 'Average BLEU Score', 'BLEU Score Distribution by Input Sequence Length', 'bleu_plot')
    plot_score_distribution(rouge_avgs, 'Average ROUGE Score', 'ROUGE Score Distribution by Input Sequence Length', 'rouge_plot')
    plot_score_distribution(meteor_avgs, 'Average METEOR Score', 'METEOR Score Distribution by Input Sequence Length', 'meteor_plot')

    overall_bleu = bleu.compute(predictions=predictions, references=[[ref] for ref in references])
    overall_rouge = rouge.compute(predictions=predictions, references=references)
    overall_meteor = meteor.compute(predictions=predictions, references=references)

    return {
        'bleu': overall_bleu['bleu'],
        'rouge': overall_rouge['rougeL'],
        'meteor': overall_meteor['meteor'],
        'score_by_length': {
            'bins': [f'{b}-{b+10}' for b in bins[:-1]],
            'avg_bleu': bleu_avgs,
            'avg_rouge': rouge_avgs,
            'avg_meteor': meteor_avgs,
            'counts': counts
        },
        'samples': [
            {'input': dataset[i]['translation']['en'],
             'prediction': predictions[i],
             'reference': references[i]}
            for i in range(min(3, len(predictions)))
        ]
    }
