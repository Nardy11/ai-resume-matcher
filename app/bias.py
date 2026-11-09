"""Gendered-word check for job descriptions.

The check works on whole word tokens, so "he" is found in "he will lead"
but not inside "the" or "help".
"""
import re

MALE_WORDS = {"he", "his", "man"}
FEMALE_WORDS = {"she", "her", "woman"}

_TOKEN = re.compile(r"\b[\w']+\b")


def tokenize(text):
    return _TOKEN.findall(text.lower())


def bias_check(text):
    tokens = set(tokenize(text))
    male = sorted(tokens & MALE_WORDS)
    female = sorted(tokens & FEMALE_WORDS)

    if len(male) > len(female):
        dominant = "male"
    elif len(female) > len(male):
        dominant = "female"
    else:
        dominant = "neutral"

    return {
        "male_bias": len(male),
        "female_bias": len(female),
        "dominant": dominant,
        "flagged_words": male + female,
    }
