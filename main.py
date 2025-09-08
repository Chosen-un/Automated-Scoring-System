# /main.py

import asyncio
import os
import json
import time
from typing import List, Dict, Any, Tuple
from langchain_core.runnables import RunnableLambda
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
import csv

from config import (
    ORIGINAL_DATA_PATH, RAG_DATA_PATH, EVAL_DATA_PATH, EXTRA_DOCS_PATH,
    PROCESSED_EVAL_DATA_PATH, GENERATED_RESPONSES_PATH, CSV_REPORT_PATH,
    TEST_DATA_PATH, TEST_RESPONSES_PATH, TEST_CSV_REPORT_PATH,
    TOP_K_TEACHER_EXAMPLES, TOP_K_FACTUAL_CHUNKS, CALL_DELAY_BETWEEN_RECORDS
)
from rag_pipeline.data_processor import (
    load_and_split_data, process_evaluation_queries, FinalAssessmentRecord,
    AssessmentRecord, RubricCriteria, QueryRewriterOutput, FeatureExtractionOutput,
    load_test_data, process_test_queries
)
from rag_pipeline.retriever import (
    prepare_rag_corpus, FAISSVectorStore, get_embedding_model,
    perform_information_retrieval, RetrievedChunk, RAGDocument
)
from rag_pipeline.generator import (
    get_scoring_and_feedback
)
from rag_pipeline.evaluator import evaluate_scoring_system

class ScoringState(BaseModel):
    """Represents the state of a single student's scoring process."""
    record: FinalAssessmentRecord
    retrieved_chunks: List[RetrievedChunk] = Field(default_factory=list)
    scoring_output: Dict[str, Any] = Field(default_factory=dict)
    
async def retrieve_chunks_node(state: ScoringState):
    """Retrieves relevant chunks and updates the state."""
    print(f"Retrieving chunks for student {state.record.student_id}...")
    updated_record = perform_information_retrieval(
        state.record,
        teacher_faiss_store,
        factual_faiss_store,
        embedding_model,
        TOP_K_TEACHER_EXAMPLES,
        TOP_K_FACTUAL_CHUNKS
    )
    state.retrieved_chunks = updated_record.retrieved_chunks
    return {"record": updated_record, "retrieved_chunks": state.retrieved_chunks}

async def generate_response_node(state: ScoringState):
    """Calls the LLM to get the score and feedback."""
    print(f"Generating score and feedback for student {state.record.student_id}...")
    scoring_output, formatted_prompt = await get_scoring_and_feedback(state.record)

    state.scoring_output = scoring_output

    state.record.predicted_score = scoring_output.get("predicted_score")
    state.record.generated_feedback = scoring_output.get("generated_feedback")
    state.record.generated_rationale = scoring_output.get("generated_rationale")
    state.record.generated_prompt_for_scoring = formatted_prompt
    
    return {"record": state.record, "scoring_output": state.scoring_output}


def create_csv_report(records: List[FinalAssessmentRecord], csv_path: str):
    """Creates a CSV report with specific fields from the final records."""
    if not records:
        print("No records to write to CSV.")
        return

    fieldnames = [
        "student_id",
        "original_question",
        "student_answer",
        "predicted_score",
        "generated_rationale",
        "generated_feedback"
    ]

    try:
        with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for record in records:
                writer.writerow({
                    "student_id": record.student_id,
                    "original_question": record.question,
                    "student_answer": record.answer,
                    "predicted_score": record.predicted_score,
                    "generated_rationale": record.generated_rationale,
                    "generated_feedback": record.generated_feedback
                })
        print(f"\nCSV report saved to: {csv_path}")
    except Exception as e:
        print(f"Error writing CSV report: {e}")

async def main():
    print("--- Starting Automated Scoring System for Teacher one.json ---")

    if not os.path.exists(RAG_DATA_PATH) or not os.path.exists(EVAL_DATA_PATH):
        print("Files not found. Starting data splitting.")
        rag_data_initial, evaluation_data_initial = load_and_split_data(ORIGINAL_DATA_PATH)
    else:
        print("Reusing existing data files.")
        with open(RAG_DATA_PATH, "r", encoding="utf-8") as f:
            rag_data_initial = [AssessmentRecord(**rec) for rec in json.load(f)]
        with open(EVAL_DATA_PATH, "r", encoding="utf-8") as f:
            evaluation_data_initial = [AssessmentRecord(**rec) for rec in json.load(f)]

    if not os.path.exists(PROCESSED_EVAL_DATA_PATH):
        processed_evaluation_data = process_evaluation_queries(evaluation_data_initial)
    else:
        print("Reusing existing processed evaluation data.")
        with open(PROCESSED_EVAL_DATA_PATH, "r", encoding="utf-8") as f:
            processed_evaluation_data = [FinalAssessmentRecord(**rec) for rec in json.load(f)]
    
    print("\nBuilding vector stores...")
    global teacher_faiss_store, factual_faiss_store, embedding_model
    teacher_rag_documents, factual_rag_documents = prepare_rag_corpus(rag_data_initial, EXTRA_DOCS_PATH)

    embedding_model = get_embedding_model()
    if not teacher_rag_documents:
        print("No teacher examples found. Cannot build teacher FAISS index.")
        return

    teacher_embedding_dimension = len(teacher_rag_documents[0].embedding)
    teacher_faiss_store = FAISSVectorStore(teacher_embedding_dimension)
    teacher_faiss_store.add_documents(teacher_rag_documents)

    factual_faiss_store = None
    if factual_rag_documents:
        factual_embedding_dimension = len(factual_rag_documents[0].embedding)
        factual_faiss_store = FAISSVectorStore(factual_embedding_dimension)
        factual_faiss_store.add_documents(factual_rag_documents)

    print("\nRunning LangGraph pipeline for scoring...")
    workflow = StateGraph(ScoringState)
    workflow.add_node("retrieve", retrieve_chunks_node)
    workflow.add_node("generate_response", generate_response_node)
    
    workflow.add_edge("retrieve", "generate_response")
    workflow.add_edge("generate_response", END)
    
    workflow.set_entry_point("retrieve")
    app = workflow.compile()

    final_responses = []
    for record in processed_evaluation_data:
        try:
            final_state = await app.ainvoke(ScoringState(record=record))
            final_responses.append(final_state["record"])
        except Exception as e:
            print(f"Error processing record {record.student_id}: {e}")
            final_responses.append(record)
        
        time.sleep(CALL_DELAY_BETWEEN_RECORDS)

    os.makedirs(os.path.dirname(GENERATED_RESPONSES_PATH), exist_ok=True)
    with open(GENERATED_RESPONSES_PATH, "w", encoding="utf-8") as f:
        json.dump([rec.model_dump() for rec in final_responses], f, indent=2)

    print(f"\n--- Automated Scoring System Complete ---")
    print(f"All generated responses saved to {GENERATED_RESPONSES_PATH}")
    create_csv_report(final_responses, CSV_REPORT_PATH)
    evaluate_scoring_system(GENERATED_RESPONSES_PATH)
    
    print("\n--- Starting Scoring for test.json ---")
    test_data = load_test_data(TEST_DATA_PATH)
    processed_test_data = process_test_queries(test_data)
    
    test_responses = []
    for record in processed_test_data:
        try:
            final_state = await app.ainvoke(ScoringState(record=record))
            test_responses.append(final_state["record"])
        except Exception as e:
            print(f"Error processing test record {record.student_id}: {e}")
            test_responses.append(record)
        
        time.sleep(CALL_DELAY_BETWEEN_RECORDS)

    os.makedirs(os.path.dirname(TEST_RESPONSES_PATH), exist_ok=True)
    with open(TEST_RESPONSES_PATH, "w", encoding="utf-8") as f:
        json.dump([rec.model_dump() for rec in test_responses], f, indent=2)
    
    print(f"\n--- Test Evaluation Complete ---")
    print(f"All test responses saved to {TEST_RESPONSES_PATH}")
    create_csv_report(test_responses, TEST_CSV_REPORT_PATH)
    
if __name__ == "__main__":
    asyncio.run(main())