import json
import re
from pathlib import Path

from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .preprocessing import clean_text

# Load model once
model = SentenceTransformer("all-MiniLM-L6-v2")

# The skill list lives in skills.json so the browser demo uses the same list.
with open(Path(__file__).with_name("skills.json"), encoding="utf-8") as f:
    SKILLS = set(json.load(f))

SKILL_THRESHOLD = 0.4


def sbert_similarity(text1, text2):
    emb = model.encode([text1, text2])
    score = float(cosine_similarity([emb[0]], [emb[1]])[0][0])
    return score


def tfidf_similarity(text1, text2):
    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform([text1, text2])
    score = float(cosine_similarity(tfidf[0], tfidf[1])[0][0])
    return score


def split_chunks(text):
    """Split raw text into lines / sentences, cleaned, empty ones dropped."""
    chunks = (clean_text(c).strip() for c in re.split(r"[\n.;!?]+", text))
    return [c for c in chunks if c]


def mentions(text, skill):
    """True if the cleaned skill name appears in the cleaned text as whole words."""
    words = clean_text(skill).split()
    pattern = r"(?<!\w)" + r"\s+".join(map(re.escape, words)) + r"(?!\w)"
    return re.search(pattern, clean_text(text)) is not None


def semantic_skill_match(text, skills=SKILLS, threshold=SKILL_THRESHOLD):
    """Skills a text covers.

    A skill counts when the text names it (whole words), or when any line or
    sentence of the text is within cosine `threshold` of the skill name
    (so "trained CNNs in PyTorch" can count as deep learning).

    Comparing the skill with the whole text instead of its sentences misses
    most skills in a CV of normal length: one embedding of 300 words is close
    to no single skill.
    """
    skills = sorted(skills)
    matched = {s for s in skills if mentions(text, s)}

    chunks = split_chunks(text)
    if chunks:
        chunk_emb = model.encode(chunks, convert_to_tensor=True)
        skill_emb = model.encode(skills, convert_to_tensor=True)
        best = util.cos_sim(chunk_emb, skill_emb).max(dim=0).values
        matched |= {s for s, score in zip(skills, best.tolist()) if score > threshold}

    return matched


def skill_gap(cv_text, job_description):
    cv_skills = semantic_skill_match(cv_text)
    job_skills = semantic_skill_match(job_description)
    return cv_skills & job_skills, job_skills - cv_skills


def combine_scores(sbert_score, tfidf_score, matched, missing):
    skill_score = len(matched) / (len(matched) + len(missing) + 1e-5)
    return 0.5 * sbert_score + 0.2 * tfidf_score + 0.3 * skill_score


def explain(matched, missing):
    if not matched and not missing:
        return "The job description names none of the tracked skills."
    if len(missing) == 0:
        return "Your CV covers all major required skills."
    if len(matched) > len(missing):
        return "You match most requirements but can improve in some areas."
    return "Your CV is missing several key skills for this role."


def advanced_score(cv_text, job_description, sbert_score):
    matched, missing = skill_gap(cv_text, job_description)
    return combine_scores(sbert_score, tfidf_similarity(cv_text, job_description), matched, missing)


def explain_result(cv_text, job_description):
    return explain(*skill_gap(cv_text, job_description))
