---
title: AI Resume Matcher
emoji: 📄
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: 6.31.0
app_file: app.py
pinned: false
license: mit
short_description: Match a CV to a job description with SBERT + TF-IDF
---

# AI Resume Matcher (Hugging Face Space)

A Gradio front end for [Nardy11/ai-resume-matcher](https://github.com/Nardy11/ai-resume-matcher).
It runs the same Python code as the FastAPI service: SBERT (`all-MiniLM-L6-v2`) semantic
similarity, TF-IDF keyword overlap, semantic skill matching against a fixed list of 16 skills,
and a whole-word check for gendered words in the job description.

## Files on the Space

```
app.py            # this Gradio app
requirements.txt
README.md         # this file (the YAML header configures the Space)
matcher/          # a copy of the repository's app/ package
```

`matcher/` is not stored in `space/`; copy it in when you deploy (see the main README,
"Deploy the Hugging Face Space").
