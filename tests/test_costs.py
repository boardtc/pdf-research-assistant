from pdf_research_assistant.costs import format_cost


def test_format_cost_gives_dollars_to_four_places():
    assert format_cost(0.0123) == "$0.0123"
    assert format_cost(0.00005) == "$0.0001"


def test_format_cost_says_when_a_real_cost_would_round_to_zero():
    # A local model's question: only the OpenAI embedding of the question is paid for.
    assert format_cost(0.00000018) == "less than $0.0001"


def test_format_cost_shows_no_cost_as_zero():
    assert format_cost(0.0) == "$0.0000"
    assert format_cost(None) == "$0.0000"
