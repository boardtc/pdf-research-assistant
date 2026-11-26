from pdf_research_assistant import quote_check
from pdf_research_assistant.quote_check import PARAPHRASE_MARK, mark_unverified_quotes

# Text as pypdf reads it from Lang ch. 8: a line break mid-sentence and a word hyphenated across lines.
CHUNK = (
    "Alfie Kohn has argued that grades reduce students' interest in\n"
    "the learning itself. Rewards can under-\nmine intrinsic moti-\nvation, he writes, "
    "and students learn to chase the grade rather than the understanding."
)


def test_a_quote_found_word_for_word_is_left_alone():
    answer = 'Lang cites Kohn: "grades reduce students\' interest in the learning itself" (Lang2021 chunk 1).'

    assert mark_unverified_quotes(answer, [CHUNK]) == (answer, [])


def test_a_quote_not_in_any_passage_loses_its_quote_marks_and_is_marked():
    answer = 'Lang stresses "the need to foster intrinsic motivators rather than depend solely on grades" here.'

    marked, flagged = mark_unverified_quotes(answer, [CHUNK])

    assert marked == (
        "Lang stresses the need to foster intrinsic motivators rather than depend solely on grades "
        f"{PARAPHRASE_MARK} here."
    )
    assert flagged == ["the need to foster intrinsic motivators rather than depend solely on grades"]


def test_curly_double_quotes_are_checked_too():
    answer = "He writes “a balanced approach to motivate students effectively” in the end."

    marked, flagged = mark_unverified_quotes(answer, [CHUNK])

    assert marked == f"He writes a balanced approach to motivate students effectively {PARAPHRASE_MARK} in the end."
    assert flagged == ["a balanced approach to motivate students effectively"]


def test_curly_single_quotes_are_checked_and_an_apostrophe_inside_them_is_kept():
    chunk = "It isn't the grade that matters to students, Kohn says."
    answer = "Kohn says ‘it isn’t the grade that matters’ and more ‘made up words for this test’."

    marked, flagged = mark_unverified_quotes(answer, [chunk])

    assert marked == f"Kohn says ‘it isn’t the grade that matters’ and more made up words for this test {PARAPHRASE_MARK}."
    assert flagged == ["made up words for this test"]


def test_curly_and_straight_apostrophes_and_quotes_inside_a_quote_count_as_the_same():
    answer = "“grades reduce students’ interest in the learning itself”"

    assert mark_unverified_quotes(answer, [CHUNK])[1] == []


def test_line_breaks_hyphenation_case_and_spacing_from_the_pdf_are_ignored():
    answer = '"Rewards can  undermine Intrinsic motivation, he writes"'

    assert mark_unverified_quotes(answer, [CHUNK])[1] == []


def test_punctuation_the_answer_adds_at_the_end_of_a_quote_is_ignored():
    answer = '"the learning itself. Rewards can undermine intrinsic motivation."'

    assert mark_unverified_quotes(answer, [CHUNK])[1] == []


def test_a_quote_with_a_different_word_is_marked():
    answer = '"Rewards can weaken intrinsic motivation"'

    assert mark_unverified_quotes(answer, [CHUNK])[1] == ["Rewards can weaken intrinsic motivation"]


def test_an_ellipsis_joins_parts_found_in_order_in_one_passage():
    for gap in ("...", "…", ". . .", "[...]"):
        answer = f'"grades reduce students\' interest {gap} chase the grade rather than the understanding"'

        assert mark_unverified_quotes(answer, [CHUNK])[1] == [], gap


def test_an_ellipsis_with_parts_out_of_order_is_marked():
    answer = '"chase the grade rather than the understanding ... grades reduce students\' interest"'

    assert len(mark_unverified_quotes(answer, [CHUNK])[1]) == 1


def test_an_ellipsis_with_parts_from_two_different_passages_is_marked():
    first, second = "Grades reduce interest in learning.", "Students chase the grade instead."
    answer = '"grades reduce interest in ... students chase the grade"'

    assert len(mark_unverified_quotes(answer, [first, second])[1]) == 1


def test_a_bracketed_insertion_counts_as_a_gap():
    answer = '"Rewards can undermine [students\'] intrinsic motivation"'

    assert mark_unverified_quotes(answer, ["Rewards can undermine intrinsic motivation, he writes."])[1] == []


def test_a_short_quoted_term_is_left_alone_even_when_not_in_the_passages():
    answer = 'Lang calls this "small teaching" and the "grade chasing habit" of students.'

    assert mark_unverified_quotes(answer, [CHUNK]) == (answer, [])


def test_a_quote_of_exactly_the_minimum_length_is_checked():
    words = " ".join(["word"] * quote_check.MIN_WORDS)

    assert mark_unverified_quotes(f'"{words}"', [CHUNK])[1] == [words]


def test_a_quote_made_only_of_gaps_and_punctuation_is_left_alone():
    answer = 'He trails off "... [...] ..." and stops.'

    assert mark_unverified_quotes(answer, [CHUNK]) == (answer, [])


def test_a_quote_found_in_any_passage_the_question_read_is_left_alone():
    answer = '"chase the grade rather than the understanding"'

    assert mark_unverified_quotes(answer, ["Something else entirely.", CHUNK])[1] == []


def test_with_no_passages_every_long_quote_is_marked():
    answer = '"chase the grade rather than the understanding"'

    assert mark_unverified_quotes(answer, [])[1] == ["chase the grade rather than the understanding"]


def test_quote_marks_never_pair_across_paragraphs():
    answer = 'He said "first part of a quote that never closes\n\nNew paragraph with "a real quote of words" in it.'

    marked, flagged = mark_unverified_quotes(answer, [CHUNK])

    assert flagged == ["a real quote of words"]
    assert marked.startswith('He said "first part of a quote that never closes\n\nNew paragraph with a real quote')


def test_a_quote_split_over_one_line_break_is_checked():
    answer = '"grades reduce students\' interest\nin the learning itself"'

    assert mark_unverified_quotes(answer, [CHUNK])[1] == []


def test_several_quotes_are_each_checked_and_only_the_missing_ones_marked():
    answer = (
        '"grades reduce students\' interest in the learning itself" but not "this sentence is not in the text" '
        'nor "neither is this one at all".'
    )

    marked, flagged = mark_unverified_quotes(answer, [CHUNK])

    assert flagged == ["this sentence is not in the text", "neither is this one at all"]
    assert marked == (
        '"grades reduce students\' interest in the learning itself" but not this sentence is not in the text '
        f"{PARAPHRASE_MARK} nor neither is this one at all {PARAPHRASE_MARK}."
    )


def test_an_answer_with_no_quotes_is_returned_unchanged():
    answer = "Lang says grades can undermine motivation (Lang2021 chunk 1)."

    assert mark_unverified_quotes(answer, [CHUNK]) == (answer, [])


def test_pdf_ligatures_match_plain_letters():
    assert mark_unverified_quotes('"the first finding of the study"', ["The ﬁrst ﬁnding of the study was"])[1] == []
