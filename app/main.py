import time

from fastapi import FastAPI, File, Form, UploadFile

from .analysis import analyze_texts
from .ingestion import extract_text_from_pdf
from .logger import log_inference_time

app = FastAPI(title="AI Resume Matcher")


@app.post("/analyze")
async def analyze(cv: UploadFile = File(...), job_description: str = Form(...)):
    start = time.time()
    cv_text = extract_text_from_pdf(await cv.read())
    result = analyze_texts(cv_text, job_description)
    log_inference_time(time.time() - start)
    return result
