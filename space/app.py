"""Gradio front end for the AI Resume Matcher, for a Hugging Face Space.

It calls the same Python code as the FastAPI service (app/analysis.py).
On the Space, that package is copied in as `matcher/` (see space/README.md);
when run from the repository it is imported from the repository root.
"""
import sys
from pathlib import Path

import gradio as gr

try:
    from matcher.analysis import analyze_texts
    from matcher.ingestion import extract_text_from_pdf
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from app.analysis import analyze_texts
    from app.ingestion import extract_text_from_pdf

EXAMPLE_CV = (
    "Machine learning engineer with 4 years of experience. Built and deployed PyTorch "
    "and TensorFlow models for demand forecasting and image classification. Strong Python, "
    "pandas and scikit-learn. Packaged models as FastAPI services in Docker and ran them on "
    "Kubernetes with CI/CD pipelines in GitHub Actions."
)
EXAMPLE_JD = (
    "We are hiring an NLP Engineer. He will build text classification and semantic search "
    "features in Python with PyTorch, deploy models with Docker and Kubernetes, and work "
    "closely with the data team on SQL pipelines. Experience with deep learning and NLP is required."
)


def run(cv_pdf, cv_text, job_description):
    if cv_pdf is not None:
        cv_text = extract_text_from_pdf(Path(cv_pdf).read_bytes())
    if not (cv_text or "").strip() or not (job_description or "").strip():
        raise gr.Error("Add a CV (PDF or text) and a job description.")

    r = analyze_texts(cv_text, job_description)
    bias = r["bias"]
    flagged = ", ".join(bias["flagged_words"]) or "none"
    summary = "\n".join(
        [
            f"### Overall score: {r['advanced_score']:.0%}",
            r["explanation"],
            "",
            "| Signal | Score |",
            "|---|---|",
            f"| Semantic similarity (SBERT) | {r['sbert_score']:.3f} |",
            f"| Keyword overlap (TF-IDF) | {r['tfidf_score']:.3f} |",
            "",
            f"**Matched skills:** {', '.join(r['matched_skills']) or 'none'}",
            "",
            f"**Missing skills:** {', '.join(r['missing_skills']) or 'none'}",
            "",
            f"**Gendered words in the job description:** {flagged} ({bias['dominant']})",
        ]
    )
    return summary, r


with gr.Blocks(title="AI Resume Matcher") as demo:
    gr.Markdown(
        "# AI Resume Matcher\n"
        "Compare a CV with a job description: SBERT semantic similarity, TF-IDF keyword overlap, "
        "matched and missing skills, and gendered words in the job ad. "
        "Source: [github.com/Nardy11/ai-resume-matcher](https://github.com/Nardy11/ai-resume-matcher)"
    )
    with gr.Row():
        with gr.Column():
            cv_pdf = gr.File(label="CV (PDF, optional)", file_types=[".pdf"], type="filepath")
            cv_text = gr.Textbox(label="...or paste the CV", lines=10, value=EXAMPLE_CV)
        with gr.Column():
            jd = gr.Textbox(label="Job description", lines=14, value=EXAMPLE_JD)
    button = gr.Button("Analyse", variant="primary")
    summary = gr.Markdown()
    raw = gr.JSON(label="Raw result (same shape as the API)")
    button.click(run, inputs=[cv_pdf, cv_text, jd], outputs=[summary, raw])

if __name__ == "__main__":
    demo.launch()
