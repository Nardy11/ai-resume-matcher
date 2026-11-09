from app.model import explain, mentions, semantic_skill_match, split_chunks


def test_named_skills_are_found_in_a_long_text():
    cv = (
        "Backend developer with five years of experience.\n"
        "Built REST services in Python and stored data in PostgreSQL with SQL.\n"
        "Shipped everything in Docker containers through a CI/CD pipeline."
    )
    skills = semantic_skill_match(cv)
    assert {"python", "sql", "docker", "ci/cd"} <= skills


def test_unrelated_text_has_no_skills():
    cv = "Head chef. Designed seasonal menus and trained junior cooks."
    assert semantic_skill_match(cv) == set()


def test_mentions_uses_whole_words():
    assert mentions("We use ML daily", "ml")
    assert not mentions("HTML and CSS", "ml")
    assert mentions("scikit-learn and pandas", "scikit-learn")


def test_split_chunks_drops_empty_lines():
    assert split_chunks("One.\n\nTwo; three!") == ["one", "two", "three"]


def test_explain_with_no_required_skills():
    assert explain(set(), set()) == "The job description names none of the tracked skills."
