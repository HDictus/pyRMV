from analysis_neuro.terminology import Term


def test_terms_added_together_makes_string():
    assert Term('pre') + Term('post') == Term('prepost')


def test_terms_added_together_combines_description():
    preterm = Term('pre',
                   "pre of {added} ")

    postterm = Term('post', "the added term")
    combined = preterm + postterm
    assert "pre of post" in combined.description
    assert postterm.description in combined.description
