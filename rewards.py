import re
import numpy as np
from typing import List


def extract_rating_from_completion(completion: str) -> int:
    """
    Extract a 0-10 integer rating from the model's completion.

    Only looks for "SCORE: X" format where X is an integer 0-10.
    Returns -1 if no valid score found.
    """
    pattern = r"SCORE:\s*(10|[0-9])\s*\Z"
    match = re.search(pattern, completion, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return -1


def check_format_compliance(completion: str) -> bool:
    """
    Check if completion follows valid format:
    - <think>...</think>\nSCORE: X (with thinking)
    - SCORE: X (no thinking)
    where X is an integer from 0-10, with no extra text after the score.
    """
    completion = completion.strip()
    pattern = r"^(<think>.*?</think>\s*)?SCORE:\s*(10|[0-9])\s*$"
    return bool(re.match(pattern, completion, re.DOTALL | re.IGNORECASE))


def compute_gaussian_reward(predicted: int, target: int, sigma: float = 2.5) -> float:
    """
    Gaussian reward function for integer scores 0-10.

    Args:
        predicted: Predicted integer score (0-10)
        target: Target integer score (0-10)
        sigma: Standard deviation (default 1.5)

    Returns:
        Reward following Gaussian distribution
    """
    distance = abs(predicted - target)
    return np.exp(-(distance**2) / (2 * sigma**2))


def format_compliance_reward_fn(
    prompts: List[str], completions: List[str], bins: List[int], **_
) -> List[float]:
    """
    Reward function for format compliance.

    Args:
        prompts: List of input prompts
        completions: List of model completions  
        bins: List of target bin numbers (0-10 integers)

    Returns:
        List of rewards (1.0 for correct format, 0.0 for incorrect format)
    """
    rewards = []
    for completion in completions:
        if check_format_compliance(completion):
            rewards.append(1.0)
        else:
            rewards.append(0.0)
    return rewards


def similarity_accuracy_reward_fn(
    prompts: List[str], completions: List[str], bins: List[int], **_
) -> List[float]:
    """
    Reward function for similarity rating accuracy.

    Args:
        prompts: List of input prompts
        completions: List of model completions
        bins: List of target bin numbers (0-10 integers)

    Returns:
        List of rewards (0.0 to 1.0) based on accuracy of similarity rating
    """
    rewards = []
    for completion, target in zip(completions, scores):
        target *= 10
        predicted_rating = extract_rating_from_completion(completion)

        if predicted_rating == -1:
            rewards.append(0.0)
            continue

        reward = compute_gaussian_reward(predicted_rating, target)
        rewards.append(reward)

    return rewards
