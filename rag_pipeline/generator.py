import json
import time
import requests
import math
import re
from typing import List, Dict, Any, Tuple, Optional
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException
from pydantic import BaseModel, Field

from config import GROQ_MODEL_NAME, GROQ_API_KEY, MAX_RETRIES, INITIAL_RETRY_DELAY, MAX_RETRY_DELAY
from rag_pipeline.data_processor import FinalAssessmentRecord, RubricCriteria

async def extract_json_from_llm_response(response: str) -> Optional[Dict]:
    """Extracts a JSON object from a string that may contain extra text."""
    try:
        match = re.search(r'{.*}', response, re.DOTALL)
        if match:
            return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return None

async def call_groq_api_with_retry(prompt_template: ChatPromptTemplate, input_data: Dict, response_schema: Optional[Dict] = None) -> Any:
    llm = ChatGroq(
        model=GROQ_MODEL_NAME,
        groq_api_key=GROQ_API_KEY,
        temperature=0.2
    )

    retry_delay = INITIAL_RETRY_DELAY
    for attempt in range(MAX_RETRIES):
        try:
            chain = prompt_template | llm
            response = (await chain.ainvoke(input_data)).content
            
            # Use the new extraction function to handle messy output
            extracted_json = await extract_json_from_llm_response(response)
            
            if extracted_json:
                return extracted_json
            else:
                raise OutputParserException("Failed to extract valid JSON from LLM response.")

        except OutputParserException as e:
            print(f"Output parsing failed on attempt {attempt+1}: {e}")
            if attempt < MAX_RETRIES - 1:
                print("Retrying with the original prompt...")
                time.sleep(retry_delay)
                retry_delay = min(retry_delay * 2, MAX_RETRY_DELAY)
            else:
                return None
        except Exception as e:
            if "rate limit" in str(e).lower() and attempt < MAX_RETRIES - 1:
                print(f"Rate limit hit. Retrying in {retry_delay:.2f} seconds...")
                time.sleep(retry_delay)
                retry_delay = min(retry_delay * 2, MAX_RETRY_DELAY)
            else:
                print(f"API call failed after {attempt+1} attempts: {e}")
                return None
    return None

def create_prompt_template(is_few_shot: bool) -> ChatPromptTemplate:
    """Creates a standardized ChatPromptTemplate with structured feedback."""
    if is_few_shot:
        template = """
        You are an expert educational AI assistant. Your task is to grade a student's answer based on the provided rubric. The students are 12 years old, so be encouraging in your grading.

        Rubric Criteria:
        {rubric_text}

        --- Factual Context (for accurate grading) ---
        {factual_context}
        --- End of Factual Context ---

        --- Example of a graded student response ---
        Question: {example_question}
        Student Answer: {example_answer}
        Rationale for Example: {example_teacher_feedback}
        Awarded Score: {example_score}
        --- End of Example ---

        Now, grade the following student's answer.
        1. Provide a score for each rubric criterion.
        2. Calculate the total score.
        3. Write a detailed, objective rationale for the teacher explaining the scores.
        4. Write a simple, encouraging, and actionable feedback for the student, using the following structure:
           ### Strengths
           * [List what the student did well, e.g., correctly defined a key term or provided a relevant example.]
           ### Areas for Improvement
           * [List one or two specific points the student can work on, e.g., 'Elaborate on how X connects to Y.']
           ### Advice
           * [Provide a short, encouraging piece of advice, e.g., 'Keep up the great work!']

        Question: {question}
        Student Answer: {answer}

        Output your response as a single JSON object wrapped in `<json>` and `</json>` tags.
        'score_breakdown': A dictionary with scores for each rubric criterion (e.g., '{{"Criterion A": 2, "Criterion B": 1}}').
        'total_score': An integer summing the scores.
        'rationale': A string with the detailed reasoning and advice for the teacher.
        'feedback': A structured string with the student's strengths, areas for improvement, and encouraging advice.
        """
        prompt_template = ChatPromptTemplate.from_template(template)
    else:
        template = """
        You are an expert educational AI assistant. Your task is to grade a student's answer based on the provided rubric. The students are 12, so be encouraging in your grading.

        Rubric Criteria:
        {rubric_text}

        --- Factual Context (for accurate grading) ---
        {factual_context}
        --- End of Factual Context ---

        Now, grade the following student's answer.
        1. Provide a score for each rubric criterion.
        2. Calculate the total score.
        3. Write a detailed, objective rationale for the teacher explaining the scores.
        4. Write a simple, encouraging, and actionable feedback for the student, using the following structure:
           ### Strengths
           * [List what the student did well, e.g., correctly defined a key term or provided a relevant example.]
           ### Areas for Improvement
           * [List one or two specific points the student can work on, e.g., 'Elaborate on how X connects to Y.']
           ### Advice
           * [Provide a short, encouraging piece of advice, e.g., 'Keep up the great work!']

        Question: {question}
        Student Answer: {answer}

        Output your response as a single JSON object wrapped in `<json>` and `</json>` tags.
        'score_breakdown': A dictionary with scores for each rubric criterion (e.g., '{{"Criterion A": 2, "Criterion B": 1}}').
        'total_score': An integer summing the scores.
        'rationale': A string with the detailed reasoning and advice for the teacher.
        'feedback': A structured string with the student's strengths, areas for improvement, and encouraging advice.
        """
        prompt_template = ChatPromptTemplate.from_template(template)

    return prompt_template

async def get_scoring_and_feedback(record: FinalAssessmentRecord) -> Tuple[Dict[str, Any], str]:
    teacher_examples = [c for c in record.retrieved_chunks if c.metadata.get("source_type") == "teacher_feedback"]
    factual_chunks = [c for c in record.retrieved_chunks if c.metadata.get("source_type") == "factual_doc"]
    
    valid_examples = [
        c for c in teacher_examples
        if c.metadata.get("original_answer") and c.metadata.get("original_answer").lower() not in ['null', 'no', 'none of the above', 'i dont know', 'i dont remember']
    ]

    best_example_chunk = None
    if valid_examples:
        sorted_examples = sorted(valid_examples, key=lambda x: x.similarity_score, reverse=True)
        if sorted_examples[0].similarity_score > -0.1:
            best_example_chunk = sorted_examples[0]
    
    rubric_to_use = record.rubric if record.rubric else await generate_rubric_from_question(record.question)
    if not rubric_to_use:
        return {"predicted_score": None, "generated_rationale": "Error: Failed to generate rubric.", "generated_feedback": "Error: Failed to generate rubric.", "score_breakdown": {}}, ""
        
    rubric_text = "\n".join([f"- {c.criteria}: Max Score {c.max_score}" for c in rubric_to_use])
    factual_context = "\n\n".join([c.content for c in factual_chunks])

    input_data = {
        "rubric_text": rubric_text,
        "factual_context": factual_context,
        "question": record.question,
        "answer": record.answer
    }

    if best_example_chunk:
        prompt_template = create_prompt_template(is_few_shot=True)
        input_data.update({
            "example_question": best_example_chunk.metadata.get("original_question", "N/A"),
            "example_answer": best_example_chunk.metadata.get("original_answer", "N/A"),
            "example_score": best_example_chunk.metadata.get("total_score", "N/A"),
            "example_teacher_feedback": best_example_chunk.metadata.get("feedback", "N/A")
        })
        print(f"Generated FEW-SHOT prompt for student {record.student_id}.")
    else:
        prompt_template = create_prompt_template(is_few_shot=False)
        print(f"Generated ZERO-SHOT prompt for student {record.student_id}. (No suitable example found)")

    response_schema = {
        "type": "OBJECT",
        "properties": {
            "score_breakdown": {
                "type": "OBJECT",
                "additionalProperties": {"type": "INTEGER"}
            },
            "total_score": {"type": "INTEGER"},
            "rationale": {"type": "STRING"},
            "feedback": {"type": "STRING"}
        },
        "required": ["score_breakdown", "total_score", "rationale", "feedback"]
    }
    
    llm_response = await call_groq_api_with_retry(prompt_template, input_data, response_schema)
    
    formatted_prompt = prompt_template.invoke(input_data).to_string()
    
    if llm_response and isinstance(llm_response, dict):
        total_score = llm_response.get("total_score")
        
        if isinstance(total_score, float):
            total_score = math.ceil(total_score)
            
        score_breakdown = llm_response.get("score_breakdown", {})
        rationale = llm_response.get("rationale")
        feedback = llm_response.get("feedback")

        rationale_str = rationale if isinstance(rationale, str) else ""
        feedback_str = feedback if isinstance(feedback, str) else ""
        
        final_response = {
            "predicted_score": total_score,
            "generated_feedback": feedback_str,
            "generated_rationale": json.dumps(score_breakdown, indent=2) + "\n\n" + rationale_str,
            "score_breakdown": score_breakdown
        }
        return final_response, formatted_prompt
    else:
        print(f"Warning: LLM response is not a valid dictionary or None: {llm_response}")
        return {"predicted_score": None, "generated_rationale": "Error: Invalid LLM response", "generated_feedback": "Error: Invalid LLM response", "score_breakdown": {}}, formatted_prompt

async def generate_rubric_from_question(question: str) -> List[RubricCriteria]:
    """Generates a generic rubric if one is not provided."""
    prompt = f"""
    You are an expert educational AI assistant.
    Given the following question, generate a rubric with 4 criteria for grading a student's answer.
    The criteria should be logical and relevant to the question. For each criterion, provide a brief name and a max score.
    Question: {question}
    Output as a JSON array of objects, each with 'criteria' (string) and 'max_score' (integer).
    """
    
    response_schema = {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "criteria": {"type": "string"},
                "max_score": {"type": "integer"}
            }
        }
    }
    
    llm_response = await call_groq_api_with_retry(ChatPromptTemplate.from_messages([("human", prompt)]), {}, response_schema)
    if llm_response:
        try:
            return [RubricCriteria(**c) for c in llm_response]
        except (ValueError, TypeError) as e:
            print(f"Error parsing generated rubric: {e}")
            return []
    return []