import re
from typing import List, Dict, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

HR_STOP_WORDS = [
    'candidate', 'candidates', 'preferred', 'prior', 'experience', 'duty', 
    'duties', 'responsibility', 'responsibilities', 'opportunity', 'employer', 
    'requirements', 'job', 'work', 'working', 'ability', 'role', 'skills',
    'years', 'year', 'location', 'position', 'company', 'team', 'help'
]
class NLPEvaluator:
    """
    Evaluates resume relevance against job descriptions using Classical NLP:
    - TF-IDF Vectorization & Cosine Similarity for Match Scoring
    - Multi-word N-gram overlap for Keyword Gap Analysis
    - Term Frequency counts for both Job Description and Resume
    - Rule-based Heuristic Feedback Generation
    """
    def __init__(self, max_features: int = 1000):
        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            ngram_range=(1, 2),
            max_features=max_features
        )

    def extract_keywords(self, text: str, top_n: int = 10) -> List[str]:
        """Extracts top N TF-IDF terms from a single text document."""
        tfidf_matrix = self.vectorizer.fit_transform([text])
        feature_names = self.vectorizer.get_feature_names_out()
        scores = tfidf_matrix.toarray()[0]
        
        sorted_indices = scores.argsort()[::-1]
        top_terms = [feature_names[i] for i in sorted_indices[:top_n] if scores[i] > 0]
        return top_terms

    def evaluate_resume(
        self, 
        resume_text: str, 
        job_title: str, 
        job_description: str, 
        top_k_keywords: int = 10
    ) -> Dict[str, Any]:
        """
        Performs vector match, term frequency tracking, keyword gap analysis,
        and recommendation generation.
        """
        # 1. TF-IDF Vectorization & Cosine Similarity Calculation
        tfidf_matrix = self.vectorizer.fit_transform([job_description, resume_text])
        similarity_score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        match_percentage = round(float(similarity_score) * 100, 2)

        # 2. Extract Top Key Domain Terms from Job Description
        feature_names = self.vectorizer.get_feature_names_out()
        job_scores = tfidf_matrix.toarray()[0]
        sorted_job_indices = job_scores.argsort()[::-1]
        
        # Filter terms to keep top non-zero TF-IDF keywords
        job_keywords = [
            (feature_names[i], round(float(job_scores[i]), 4)) 
            for i in sorted_job_indices 
            if job_scores[i] > 0
        ][:top_k_keywords]

        # 3. Text Preprocessing for Exact Match Scanning
        resume_clean = resume_text.lower()
        resume_clean = re.sub(r'[^a-z0-9\s]', ' ', resume_clean)
        resume_clean = re.sub(r'\s+', ' ', resume_clean).strip()

        job_clean = job_description.lower()
        job_clean = re.sub(r'[^a-z0-9\s]', ' ', job_clean)
        job_clean = re.sub(r'\s+', ' ', job_clean).strip()

        # 4. Detailed Term Overlap & Frequency Breakdown
        matching_details = []
        matching_keywords = []
        missing_keywords = []

        for kw, score in job_keywords:
            kw_clean = kw.strip()
            # Non-strangling regex pattern for flexible phrase matching
            pattern = r'(?<![a-z0-9])' + re.escape(kw_clean) + r'(?![a-z0-9])'
            
            resume_matches = len(re.findall(pattern, resume_clean))
            job_matches = len(re.findall(pattern, job_clean))

            if resume_matches > 0:
                matching_keywords.append(kw_clean)
                matching_details.append({
                    "keyword": kw_clean,
                    "tfidf_weight": score,
                    "resume_occurrences": resume_matches,
                    "job_occurrences": job_matches
                })
            else:
                missing_keywords.append(kw_clean)

        # 5. Compute Coverage Metrics & Categorical Match Level
        total_keywords_analyzed = len(job_keywords)
        keyword_coverage = (
            round((len(matching_keywords) / total_keywords_analyzed) * 100, 2)
            if total_keywords_analyzed > 0 else 0.0
        )

        if match_percentage >= 70:
            match_tier = "High Alignment"
        elif match_percentage >= 45:
            match_tier = "Moderate Alignment"
        else:
            match_tier = "Low Alignment"

        # 6. Generate Rule-Based Feedback Recommendations
        recommendations = []
        if match_percentage < 50:
            recommendations.append(
                f"Low overall match ({match_percentage}%). Consider tailoring bullet points directly to core responsibilities of {job_title}."
            )
        else:
            recommendations.append(f"Good structural alignment detected ({match_percentage}% match score).")

        if missing_keywords:
            recommendations.append(
                f"Missing critical domain terms: {', '.join(missing_keywords[:5])}. Integrate these naturally into your technical skills section."
            )

        return {
            "match_percentage": match_percentage,
            "keyword_coverage_percentage": keyword_coverage,
            "match_tier": match_tier,
            "matching_keywords_count": len(matching_keywords),
            "total_job_keywords_analyzed": total_keywords_analyzed,
            "matching_keywords_details": matching_details,
            "missing_keywords": missing_keywords,
            "recommendations": recommendations,
            "evaluation_method": "TF-IDF + Cosine Similarity (Enhanced Classical NLP)"
        }