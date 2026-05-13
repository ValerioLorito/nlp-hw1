import nltk
from nltk.translate.meteor_score import meteor_score

def compute_exact_match(prediction, ground_truths):
    for ground_truth in ground_truths:
        if prediction.strip().lower() == ground_truth.strip().lower():
            return 1
    return 0

def compute_sub_EM(prediction, ground_truths):
    for ground_truth in ground_truths:
        if ground_truth.strip().lower() in prediction.strip().lower():
            return 1
    return 0

def compute_meteor(prediction, ground_truths):
    scores = []
    prediction_tokens = nltk.word_tokenize(prediction.lower())
    for ground_truth in ground_truths:
        ground_truth_tokens = nltk.word_tokenize(ground_truth.lower())
        scores.append(meteor_score([ground_truth_tokens], prediction_tokens))
    return max(scores) if scores else 0

def evaluate_all(prediction, ground_truth): # prediction is a string and ground_truth is a list !!!! (but composed of 1 element)
    return {
        "EM" : (compute_exact_match(prediction, ground_truth)),
        "subEM" : (compute_sub_EM(prediction, ground_truth)),
        "METEOR" : (compute_meteor(prediction, ground_truth))
    }
    
