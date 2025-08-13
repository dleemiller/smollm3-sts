import pytest
from rewards import (
    extract_rating_from_completion,
    check_format_compliance,
    compute_gaussian_reward,
    format_compliance_reward_fn,
    similarity_accuracy_reward_fn,
)


@pytest.mark.parametrize("completion,expected", [
    # Valid integer scores
    ("SCORE: 0", 0.0),
    ("SCORE: 5", 5.0),
    ("SCORE: 10", 10.0),
    ("SCORE:8", 8.0),
    ("score: 3", 3.0),
    ("Some text\nSCORE: 7", 7.0),
    
    # Valid decimal scores
    ("SCORE: 0.0", 0.0),
    ("SCORE: 5.5", 5.5),
    ("SCORE: 10.0", 10.0),
    ("SCORE: 7.3", 7.3),
    ("SCORE:8.9", 8.9),
    ("score: 3.2", 3.2),
    ("SCORE: 0.1", 0.1),
    ("SCORE: 9.9", 9.9),
    
    # Invalid cases (should return -1.0)
    ("SCORE: 11", -1.0),
    ("SCORE: 10.1", -1.0),
    ("SCORE: -1", -1.0),
    ("SCORE: 5.55", -1.0),
    ("SCORE: 5.67", -1.0),  # Too many decimal places
    ("SCORE: 3.12", -1.0),  # Too many decimal places
    ("SCORE: 8.05", -1.0),  # Too many decimal places
    ("SCORE: 2.94", -1.0),  # Too many decimal places
    ("Rating: 5", -1.0),
    ("SCORE:", -1.0),
    ("SCORE: abc", -1.0),
    ("", -1.0),
    ("SCORE: 7\nExtra text", -1.0),
    ("SCORE: 5.0 with extra", -1.0),
])
def test_extract_rating(completion, expected):
    assert extract_rating_from_completion(completion) == expected


@pytest.mark.parametrize("completion,expected", [
    # Valid formats
    ("<think>reasoning</think>\nSCORE: 7.5", True),
    ("SCORE: 5.0", True),
    ("<think></think>\nSCORE: 3.9", True),
    ("SCORE:8.2", True),
    ("score: 0.0", True),
    ("<think>analysis</think>\nSCORE: 10", True),
    ("SCORE: 0", True),
    ("SCORE: 9.9", True),
    
    # Invalid formats
    ("Rating: 5", False),
    ("SCORE: 11.0", False),
    ("SCORE: 5.55", False),
    ("<think>reasoning</think>\nSCORE: 7.5\nExtra", False),
    ("SCORE: 7.0 with extra", False),
    ("<think>Incomplete", False),
    ("think>No opening</think>\nSCORE: 5.0", False),
    ("", False),
    ("No score", False),
    ("SCORE: -1.5", False),
])
def test_format_compliance(completion, expected):
    assert check_format_compliance(completion) == expected


@pytest.mark.parametrize("predicted,target,expected_perfect", [
    (5.0, 5.0, True),
    (0.0, 0.0, True), 
    (10.0, 10.0, True),
    (3.7, 3.7, True),
    (5.0, 6.0, False),
    (7.5, 5.0, False),
])
def test_gaussian_reward(predicted, target, expected_perfect):
    reward = compute_gaussian_reward(predicted, target)
    if expected_perfect:
        assert reward == 1.0
    else:
        assert 0.0 < reward < 1.0


def test_reward_functions():
    """Test both reward functions with comprehensive examples"""
    prompts = ["test"] * 8
    completions = [
        "<think>perfect</think>\nSCORE: 7.5",     # valid format, perfect score
        "SCORE: 7.0",                             # valid format, close score  
        "SCORE: 5.0",                             # valid format, distant score
        "Rating: 7.5",                            # invalid format
        "SCORE: 11.0",                            # invalid score
        "<think>good</think>\nSCORE: 0.0",        # valid format, perfect score
        "SCORE: 2.55",                            # invalid format (too many decimals)
        "SCORE: 8.5",                             # valid format, close score
    ]
    targets = [7.5, 7.5, 7.5, 7.5, 7.5, 0.0, 2.5, 8.0]
    
    # Test format compliance
    format_rewards = format_compliance_reward_fn(prompts, completions, targets)
    expected_format = [1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0]
    assert format_rewards == expected_format
    
    # Test similarity accuracy  
    similarity_rewards = similarity_accuracy_reward_fn(prompts, completions, targets)
    
    assert similarity_rewards[0] == 1.0  # perfect match
    assert 0.5 < similarity_rewards[1] < 1.0  # close match (distance 0.5)
    assert 0.3 < similarity_rewards[2] < 0.8  # distant match (distance 2.5)
    assert similarity_rewards[3] == 0.0  # invalid format
    assert similarity_rewards[4] == 0.0  # invalid score
    assert similarity_rewards[5] == 1.0  # perfect match
    assert similarity_rewards[6] == 0.0  # invalid format
    assert 0.5 < similarity_rewards[7] < 1.0  # close match
