import json
import config
from src.data_loader import load_and_clean_data
from src.nlp_extractor import NLPEvaluator


SAMPLE_RESUME = """
Jane Doe - Business Analyst
Experience: 4 years implementing cloud hosted migration software and managing process migration.
Preferred candidates experience: Led software migration projects, worked with cloud infrastructure, and supported business teams.
Skills: Process Migration, Cloud Hosted Systems, Implementing SaaS Solutions, SQL, Requirement Gathering.
"""

def run_pipeline(target_job_index: int = 0):
    print("=" * 50)
    print("      NLP JOB MATCH & RESUME TAILORING      ")
    print("=" * 50)

    # 1. Load Data
    print("\n[1/3] Loading dataset...")
    df = load_and_clean_data(config.DATA_PATH)
    job_title = df.iloc[target_job_index]['clean_title']
    job_desc = df.iloc[target_job_index]['cleaned_description']
    print(f"✅ Selected Target Job [{target_job_index}]: {job_title}")

    # 2. Pure Classical NLP Evaluation
    print("\n[2/3] Computing Vector Similarity & Keyword Gap Analysis...")
    evaluator = NLPEvaluator(max_features=config.TFIDF_MAX_FEATURES)
    
    result = evaluator.evaluate_resume(
        resume_text=SAMPLE_RESUME,
        job_title=job_title,
        job_description=job_desc,
        top_k_keywords=config.TOP_K_KEYWORDS
    )

    # 3. Output Results
    print("\n[3/3] Pipeline Complete! Pure NLP Evaluation Results:")
    print("-" * 50)
    print(json.dumps(result, indent=2))
    print("-" * 50)

if __name__ == "__main__":
    run_pipeline(target_job_index=1)