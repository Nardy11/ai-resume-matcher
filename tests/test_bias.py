from app.bias import bias_check, tokenize


def test_the_does_not_trigger_he():
    # "the", "other", "where" and "help" all contain "he" as a substring.
    result = bias_check("the other team will help where needed")
    assert result["male_bias"] == 0
    assert result["female_bias"] == 0
    assert result["dominant"] == "neutral"
    assert result["flagged_words"] == []


def test_shell_and_human_do_not_trigger():
    # "she" in "shell", "her" in "there", "man" in "manager" / "human".
    result = bias_check("shell scripting, human resources manager, there and here")
    assert result["male_bias"] == 0
    assert result["female_bias"] == 0


def test_whole_words_are_flagged():
    result = bias_check("He will lead the team and his manager reports weekly")
    assert result["male_bias"] == 2
    assert result["dominant"] == "male"
    assert result["flagged_words"] == ["he", "his"]


def test_female_words_are_flagged():
    result = bias_check("She owns the roadmap; her team ships weekly")
    assert result["female_bias"] == 2
    assert result["dominant"] == "female"


def test_tokenize_keeps_apostrophes():
    assert tokenize("He's here") == ["he's", "here"]
