# /rag_pipeline/data_processor.py

import json
import os
import random
import re
import textstat
import nltk
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Tuple, Optional
from transformers import pipeline
from nltk.corpus import stopwords
from docx import Document as DocxDocument
from config import RAG_DATA_PATH, EVAL_DATA_PATH, PROCESSED_EVAL_DATA_PATH, DATA_DIR, ORIGINAL_DATA_PATH, EXTRA_DOCS_PATH

# Download NLTK stopwords if not already downloaded
nltk.download('stopwords', quiet=True)

class RubricCriteria(BaseModel):
    criteria: str
    max_score: int
    score_awarded: Optional[int] = None

class AssessmentRecord(BaseModel):
    student_id: str
    question: str
    answer: str
    rubric: List[RubricCriteria]
    total_score: int
    feedback: str
    overall_quality_of_answer: int

class TestAssessmentRecord(BaseModel):
    student_id: str
    question: str
    answer: str
    rubric: List[Dict[str, Any]]
    max_score: int

class QueryRewriterOutput(BaseModel):
    best_file_path: Optional[str] = None
    file_locator: str
    similarity_score: Optional[float] = None
    only_question: str

class FeatureExtractionOutput(BaseModel):
    complexity: str
    complexity_score: float
    readability: str
    readability_score: float

class FinalAssessmentRecord(BaseModel):
    student_id: str
    question: str
    answer: str
    rubric: List[RubricCriteria]
    total_score: Optional[int] = None
    feedback: Optional[str] = None
    overall_quality_of_answer: Optional[int] = None
    query_rewriter: List[QueryRewriterOutput] = Field(default_factory=list)
    feature_extraction: List[FeatureExtractionOutput] = Field(default_factory=list)
    retrieved_chunks: List[Any] = Field(default_factory=list)
    generated_prompt_for_scoring: Optional[str] = None
    predicted_score: Optional[int] = None
    generated_feedback: Optional[str] = None
    generated_rationale: Optional[str] = None

_sentiment_classifier = None
_ner_model = None

def get_sentiment_classifier():
    global _sentiment_classifier
    if _sentiment_classifier is None:
        print("Loading sentiment classifier model...")
        _sentiment_classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    return _sentiment_classifier

def get_ner_model():
    global _ner_model
    if _ner_model is None:
        print("Loading NER model...")
        _ner_model = pipeline("ner", model="Jean-Baptiste/roberta-large-ner-english", aggregation_strategy="simple")
    return _ner_model

def load_and_split_data(json_file_path: str, rag_ratio: float = 0.6) -> Tuple[List[AssessmentRecord], List[AssessmentRecord]]:
    if not os.path.exists(json_file_path):
        print(f"Error: Original data file not found at {json_file_path}")
        return [], []

    with open(json_file_path, "r", encoding="utf-8") as f:
        all_records_raw = json.load(f)

    all_records = [AssessmentRecord(**record) for record in all_records_raw]
    random.seed(42)
    random.shuffle(all_records)

    split_point = int(len(all_records) * rag_ratio)
    rag_data = all_records[:split_point]
    evaluation_data = all_records[split_point:]

    os.makedirs(DATA_DIR, exist_ok=True)
    with open(RAG_DATA_PATH, "w", encoding="utf-8") as f:
        json.dump([rec.model_dump() for rec in rag_data], f, indent=2)
    with open(EVAL_DATA_PATH, "w", encoding="utf-8") as f:
        json.dump([rec.model_dump() for rec in evaluation_data], f, indent=2)

    print(f"Data loaded and split: {len(rag_data)} records for RAG, {len(evaluation_data)} for evaluation.")
    return rag_data, evaluation_data

def load_and_process_docx(file_path: str) -> List[Dict]:
    if not os.path.exists(file_path):
        print(f"Warning: DOCX file not found at {file_path}. Skipping.")
        return []
    
    print(f"Loading and processing DOCX file: {file_path}")
    doc = DocxDocument(file_path)
    
    extra_docs = []
    
    for para in doc.paragraphs:
        text = para.text.strip()
        if not text:
            continue
            
        source_match = re.search(r'\\', text)
        source_numbers = None
        if source_match:
            source_numbers = source_match.group(1)
            clean_text = text.replace(source_match.group(0), "").strip()
        else:
            clean_text = text

        if clean_text:
            extra_docs.append({
                "id": len(extra_docs) + 1,
                "text": clean_text,
                "source": source_numbers if source_numbers else "N/A",
                "metadata": {"source_type": "factual_doc"}
            })
            
    print(f"Processed {len(extra_docs)} extra factual documents from DOCX.")
    return extra_docs

def calculate_readability(text: str) -> Tuple[str, float]:
    if not text.strip():
        return "unknown", 0.0
    try:
        dale_chall = textstat.dale_chall_readability_score(text)
    except Exception:
        return "unknown", 0.0

    if dale_chall < 8.0:
        return "non-expert", round(dale_chall, 2)
    else:
        return "expert", round(dale_chall, 2)

def calculate_complexity(query: str) -> Tuple[str, float]:
    if not query.strip():
        return "unknown", 0.0
    try:
        classifier = get_sentiment_classifier()
        result = classifier(query)[0]
        label = result['label']
        score = result['score']
        if label == 'POSITIVE':
            return "verbose", round(score, 4)
        elif label == 'NEGATIVE':
            return "vague", round(score, 4)
        else:
            return "unknown", round(score, 4)
    except Exception as e:
        print(f"Error calculating complexity: {e}. Returning unknown.")
        return "unknown", 0.0

def split_question_ner(query: str) -> Tuple[str, str]:
    unwanted_words = ['agreement', 'nda', 'non-disclosure', 'agreements', 'content', 'co-branding', 'license', "acquisition", "merger"]

    parts = query.split(";", 1)
    if len(parts) == 2:
        targeted_corpus = parts[0].replace("Consider", "").strip()
        stop_words = set(stopwords.words("english"))
        tokens = targeted_corpus.split()
        filtered_tokens = [token for token in tokens if token.lower() not in stop_words]
        targeted_corpus = " ".join(filtered_tokens).strip()
        original_question = parts[1].strip()
        return targeted_corpus, original_question
    else:
        try:
            ner_model = get_ner_model()
            ner_results = ner_model(query)

            org_entities = [
                ent["word"].strip() for ent in ner_results
                if ent.get("entity_group") in ["ORG", "MISC"] and ent.get("score", 0) > 0.8
            ]

            pattern = re.compile(r"\\b(" + "|".join(unwanted_words) + r")\\b", re.IGNORECASE)
            filtered_orgs = [pattern.sub("", org).strip() for org in org_entities]
            filtered_orgs = [org for org in filtered_orgs if org]

            if filtered_orgs:
                targeted_corpus = " ".join(filtered_orgs)
            else:
                targeted_corpus = ""

            original_question = query
            if targeted_corpus:
                original_question = re.sub(re.escape(targeted_corpus), "", original_question, flags=re.IGNORECASE).strip()
                original_question = re.sub(r"^Consider\\s+", "", original_question, flags=re.IGNORECASE).strip()
                original_question = re.sub(r"^[;,.!?\\s]+\\s*", "", original_question).strip()
                original_question = re.sub(r"[;,.!?\\s]+$", "", original_question).strip()

            return targeted_corpus, original_question
        except Exception as e:
            print(f"Error during NER-based split: {e}. Falling back to full query as question.")
            return "", query.strip()

def process_evaluation_queries(evaluation_data: List[AssessmentRecord]) -> List[FinalAssessmentRecord]:
    print("Processing evaluation queries: query translation and feature extraction...")
    processed_records = []

    for record in evaluation_data:
        targeted_corpus, only_question = split_question_ner(record.question)
        readability_category, readability_score = calculate_readability(only_question if only_question else record.question)
        complexity_category, complexity_score = calculate_complexity(only_question if only_question else record.question)

        processed_record = FinalAssessmentRecord(
            student_id=record.student_id,
            question=record.question,
            answer=record.answer,
            rubric=record.rubric,
            total_score=record.total_score,
            feedback=record.feedback,
            overall_quality_of_answer=record.overall_quality_of_answer,
            query_rewriter=[QueryRewriterOutput(
                file_locator=targeted_corpus,
                only_question=only_question
            )],
            feature_extraction=[FeatureExtractionOutput(
                complexity=complexity_category,
                complexity_score=complexity_score,
                readability=readability_category,
                readability_score=readability_score
            )]
        )
        processed_records.append(processed_record)

    with open(PROCESSED_EVAL_DATA_PATH, "w", encoding="utf-8") as f:
        json.dump([rec.model_dump() for rec in processed_records], f, indent=2)

    print(f"Processed {len(processed_records)} evaluation queries. Results saved to {PROCESSED_EVAL_DATA_PATH}")
    return processed_records

def load_test_data(file_path: str) -> List[TestAssessmentRecord]:
    if not os.path.exists(file_path):
        print(f"Error: Test data file not found at {file_path}")
        return []

    with open(file_path, "r", encoding="utf-8") as f:
        records_raw = json.load(f)

    return [TestAssessmentRecord(**record) for record in records_raw]

def process_test_queries(test_data: List[TestAssessmentRecord]) -> List[FinalAssessmentRecord]:
    print("Processing test queries: query translation and feature extraction...")
    processed_records = []
    
    for record in test_data:
        rubric_criteria = [RubricCriteria(**c) for c in record.rubric]
        
        processed_record = FinalAssessmentRecord(
            student_id=record.student_id,
            question=record.question,
            answer=record.answer,
            rubric=rubric_criteria,
            total_score=None,
            feedback=None,
            overall_quality_of_answer=None,
            query_rewriter=[QueryRewriterOutput(file_locator="", only_question=record.question)],
            feature_extraction=[]
        )
        processed_records.append(processed_record)
        
    return processed_records