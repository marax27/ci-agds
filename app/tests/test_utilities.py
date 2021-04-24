from ..utilities import binary_search


def test_binary_search_single_element_list():
    assert binary_search([1.23], 3.14) == 0


def test_binary_search_exact_result_exists():
    assert binary_search([1, 2, 3, 4, 5, 6], 3) == 2


def test_binary_search_1():
    assert binary_search([1, 2, 3, 4, 5, 6], 3.1) == 2


def test_binary_search_2():
    assert binary_search([1, 2, 3, 4, 5, 6], 3.8) == 3


def test_binary_search_3():
    assert binary_search([1, 2, 3, 4, 5, 6], 6.2) == 5


def test_binary_search_4():
    assert binary_search([1, 2, 3, 4, 5, 6], 1.7) == 1
