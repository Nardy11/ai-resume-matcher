"""One entry point for the whole CV-vs-job-description analysis.

Used by the FastAPI service, the Hugging Face Space and the evaluation script,
so all three compute exactly the same thing. demo/matcher.js is the browser
port of this file.
"""
from .bias import bias_check
from .model import combine_scores, explain, sbert_similarity, skill_gap, tfidf_similarity
from .preprocessing import clean_text


def analyze_texts(cv_text, job_description):
    # Skills are found per line / sentence, so they get the raw text
    # (cleaning removes the full stops that separate sentences).
    matched, missing = skill_gap(cv_text, job_description)

    cv_clean = clean_text(cv_text)
    jd_clean = clean_text(job_description)
    sbert_score = sbert_similarity(cv_clean, jd_clean)
    tfidf_score = tfidf_similarity(cv_clean, jd_clean)

    return {
        "sbert_score": sbert_score,
        "tfidf_score": tfidf_score,
        "advanced_score": combine_scores(sbert_score, tfidf_score, matched, missing),
        "matched_skills": sorted(matched),
        "missing_skills": sorted(missing),
        "bias": bias_check(jd_clean),
        "explanation": explain(matched, missing),
    }
