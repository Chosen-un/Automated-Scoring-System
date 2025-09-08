# /rag_pipeline/evaluator.py

import json
import numpy as np
import os
import math
from typing import List, Optional
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer
try:
    from bert_score import score as bert_score
except ImportError:
    bert_score = None

from rag_pipeline.data_processor import FinalAssessmentRecord, RubricCriteria, QueryRewriterOutput, FeatureExtractionOutput
from rag_pipeline.retriever import get_embedding_model, RetrievedChunk

_rouge_scorer = None 

def get_rouge_scorer():
    global _rouge_scorer
    if _rouge_scorer is None:
        _rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    return _rouge_scorer

def evaluate_scoring_system(generated_responses_path: str):
    print("\n--- Starting Evaluation ---\n")
    if not os.path.exists(generated_responses_path):
        print(f"Error: Generated responses file not found at {generated_responses_path}")
        return

    with open(generated_responses_path, "r", encoding="utf-8") as f:
        records_raw = json.load(f)

    records = []
    for r in records_raw:
        predicted_score = r.get('predicted_score')
        if isinstance(predicted_score, float):
            predicted_score = math.ceil(predicted_score)
        
        # Manually reconstruct the record to handle potential missing fields
        record_obj = FinalAssessmentRecord(
            student_id=r['student_id'],
            question=r['question'],
            answer=r['answer'],
            rubric=[RubricCriteria(**crit) for crit in r.get('rubric', [])],
            total_score=r.get('total_score'),
            feedback=r.get('feedback'),
            overall_quality_of_answer=r.get('overall_quality_of_answer'),
            query_rewriter=[QueryRewriterOutput(**qr) for qr in r.get('query_rewriter', [])],
            feature_extraction=[FeatureExtractionOutput(**fe) for fe in r.get('feature_extraction', [])],
            retrieved_chunks=[RetrievedChunk(**rc) for rc in r.get('retrieved_chunks', [])],
            generated_prompt_for_scoring=r.get('generated_prompt_for_scoring'),
            predicted_score=predicted_score,
            generated_feedback=r.get('generated_feedback'),
            generated_rationale=r.get('generated_rationale')
        )
        records.append(record_obj)
    
    scorer = get_rouge_scorer()
    
    valid_records = [
        r for r in records
        if r.predicted_score is not None and
           r.total_score is not None and
           isinstance(r.generated_feedback, str) and
           r.generated_feedback != "Error: Invalid LLM response" and
           isinstance(r.feedback, str) and
           r.feedback.strip()
    ]

    if not valid_records:
        print("No valid records with ground truth data to evaluate.")
        return

    actual_scores = [r.total_score for r in valid_records]
    predicted_scores = [r.predicted_score for r in valid_records]
    actual_feedbacks = [r.feedback for r in valid_records]
    generated_feedbacks = [r.generated_feedback for r in valid_records]
    actual_rationales = [r.feedback for r in valid_records]
    generated_rationales = [r.generated_rationale for r in valid_records]

    if len(actual_scores) == 0:
        print("No records with valid scores to evaluate.")
        return

    mae = mean_absolute_error(actual_scores, predicted_scores)
    mse = mean_squared_error(actual_scores, predicted_scores)
    exact_matches = sum(1 for a, p in zip(actual_scores, predicted_scores) if a == p)
    accuracy = exact_matches / len(actual_scores)
    within_one_tolerance = sum(1 for a, p in zip(actual_scores, predicted_scores) if abs(a - p) <= 1)
    tolerance_accuracy = within_one_tolerance / len(actual_scores)

    print(f"\nScore Evaluation:")
    print(f"  Mean Absolute Error (MAE): {mae:.2f}")
    print(f"  Mean Squared Error (MSE): {mse:.2f}")
    print(f"  Exact Score Match Accuracy: {accuracy:.2f} ({exact_matches}/{len(actual_scores)})")
    print(f"  Scores within +/- 1 Tolerance: {tolerance_accuracy:.2f} ({within_one_tolerance}/{len(actual_scores)})\n")

    print(f"Textual Content Evaluation (Cosine Similarity, ROUGE, and BERTScore):")
    embedding_model = get_embedding_model()

    non_empty_actual_feedbacks = [f for f in actual_feedbacks if f and f.strip()]
    non_empty_generated_feedbacks = [f for f in generated_feedbacks if f and f.strip()]
    if non_empty_actual_feedbacks and non_empty_generated_feedbacks:
        actual_feedback_embeddings = embedding_model.encode(non_empty_actual_feedbacks)
        generated_feedback_embeddings = embedding_model.encode(non_empty_generated_feedbacks)

        min_len_feedback = min(len(actual_feedback_embeddings), len(generated_feedback_embeddings))
        if min_len_feedback > 0:
            avg_feedback_similarity = np.mean(
                [cosine_similarity(actual_feedback_embeddings[i].reshape(1, -1), generated_feedback_embeddings[i].reshape(1, -1))[0][0]
                for i in range(min_len_feedback)]
            )
            print(f"  Average Feedback Similarity (Cosine): {avg_feedback_similarity:.3f}")

            rouge_scores = [scorer.score(record.feedback, record.generated_feedback) for record in valid_records]
            avg_rouge1 = np.mean([s['rouge1'].fmeasure for s in rouge_scores])
            avg_rouge2 = np.mean([s['rouge2'].fmeasure for s in rouge_scores])
            avg_rougeL = np.mean([s['rougeL'].fmeasure for s in rouge_scores])
            print(f"  Average Feedback ROUGE-1: {avg_rouge1:.3f}")
            print(f"  Average Feedback ROUGE-2: {avg_rouge2:.3f}")
            print(f"  Average Feedback ROUGE-L: {avg_rougeL:.3f}")

            if bert_score:
                P, R, F1 = bert_score(non_empty_generated_feedbacks, non_empty_actual_feedbacks, lang="en")
                print(f"  Average Feedback BERTScore P: {P.mean().item():.3f}")
                print(f"  Average Feedback BERTScore R: {R.mean().item():.3f}")
                print(f"  Average Feedback BERTScore F1: {F1.mean().item():.3f}")

    else:
        print("  Not enough non-empty feedback texts to calculate similarity.")

    non_empty_actual_rationales = [r for r in actual_rationales if r.strip()]
    non_empty_generated_rationales = [r for r in generated_rationales if r.strip() and r != "Error: Invalid LLM response"]
    if non_empty_actual_rationales and non_empty_generated_rationales:
        actual_rationale_embeddings = embedding_model.encode(non_empty_actual_rationales)
        generated_rationale_embeddings = embedding_model.encode(non_empty_generated_rationales)

        min_len_rationale = min(len(actual_rationale_embeddings), len(generated_rationale_embeddings))

        if min_len_rationale > 0:
            avg_rationale_similarity = np.mean(
                [cosine_similarity(actual_rationale_embeddings[i].reshape(1, -1), generated_rationale_embeddings[i].reshape(1, -1))[0][0]
                for i in range(min_len_rationale)]
            )
            print(f"  Average Rationale Similarity (vs. Original Feedback): {avg_rationale_similarity:.3f}")

            rouge_scores_rationale = [scorer.score(record.feedback, record.generated_rationale) for record in valid_records]
            avg_rouge1_rationale = np.mean([s['rouge1'].fmeasure for s in rouge_scores_rationale])
            avg_rouge2_rationale = np.mean([s['rouge2'].fmeasure for s in rouge_scores_rationale])
            avg_rougeL_rationale = np.mean([s['rougeL'].fmeasure for s in rouge_scores_rationale])
            print(f"  Average Rationale ROUGE-1: {avg_rouge1_rationale:.3f}")
            print(f"  Average Rationale ROUGE-2: {avg_rouge2_rationale:.3f}")
            print(f"  Average Rationale ROUGE-L: {avg_rougeL_rationale:.3f}")
            
            if bert_score:
                P, R, F1 = bert_score(non_empty_generated_rationales, non_empty_actual_rationales, lang="en")
                print(f"  Average Rationale BERTScore P: {P.mean().item():.3f}")
                print(f"  Average Rationale BERTScore R: {R.mean().item():.3f}")
                print(f"  Average Rationale BERTScore F1: {F1.mean().item():.3f}")

    else:
        print("  Not enough non-empty rationale texts to calculate similarity.")

    print("\n--- Evaluation Complete ---\n")
    print("Note: Textual similarity is a proxy. Human evaluation is recommended for true quality assessment.")