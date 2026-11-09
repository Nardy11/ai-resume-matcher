from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_analyze():
    with open("tests/sample_cv.pdf", "rb") as f:
        response = client.post(
            "/analyze",
            files={"cv": ("sample_cv.pdf", f, "application/pdf")},
            data={"job_description": "machine learning developer"}
        )
    assert response.status_code == 200


def test_analyze_keys():
    with open("tests/sample_cv.pdf", "rb") as f:
        response = client.post(
            "/analyze",
            files={"cv": ("sample_cv.pdf", f, "application/pdf")},
            data={"job_description": "machine learning developer"}
        )

    data = response.json()

    assert "sbert_score" in data
    assert "tfidf_score" in data
    assert "advanced_score" in data
    assert "missing_skills" in data
    assert "bias" in data


def test_missing_skills_not_empty():
    with open("tests/sample_cv.pdf", "rb") as f:
        response = client.post(
            "/analyze",
            files={"cv": ("sample_cv.pdf", f, "application/pdf")},
            data={"job_description": "docker kubernetes ci cd"}
        )

    data = response.json()
    assert isinstance(data["missing_skills"], list)


def test_scores_not_null():
    with open("tests/sample_cv.pdf", "rb") as f:
        response = client.post(
            "/analyze",
            files={"cv": ("sample_cv.pdf", f, "application/pdf")},
            data={"job_description": "docker kubernetes ci cd"}
        )

    data = response.json()

    assert data["sbert_score"] is not None
    assert data["tfidf_score"] is not None
    assert data["advanced_score"] is not None
    assert data["missing_skills"] is not None
    assert data["bias"] is not None

def test_bias_ignores_he_inside_the():
    with open("tests/sample_cv.pdf", "rb") as f:
        response = client.post(
            "/analyze",
            files={"cv": ("sample_cv.pdf", f, "application/pdf")},
            data={"job_description": "the other engineer will help the team"}
        )

    bias = response.json()["bias"]
    assert bias["male_bias"] == 0
    assert bias["female_bias"] == 0
    assert bias["dominant"] == "neutral"
