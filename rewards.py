import re
import numpy as np
from typing import List


def extract_rating_from_completion(completion: str) -> float:
    """
    Extract a 0.0-10.0 rating with single digit precision from the model's completion.

    Only looks for "SCORE: X.Y" or "SCORE: X" format where X.Y is a number 0.0-10.0.
    Returns -1.0 if no valid score found.
    """
    # Match patterns like "SCORE: 5.7", "SCORE: 10.0", "SCORE: 3", etc.
    pattern = r"SCORE:\s*(10\.0|10|[0-9](?:\.[0-9])?)\s*\Z"
    match = re.search(pattern, completion, re.IGNORECASE)
    if match:
        score = float(match.group(1))
        # Ensure it's within valid range and round to single decimal precision
        if 0.0 <= score <= 10.0:
            return round(score, 1)
    return -1.0


def check_format_compliance(completion: str) -> bool:
    """
    Check if completion follows valid format:
    - <think>...</think>\nSCORE: X.Y (with thinking)
    - SCORE: X.Y (no thinking)
    where X.Y is a number from 0.0-10.0 with single digit precision,
    with no extra text after the score.
    """
    completion = completion.strip()
    # Updated pattern to accept decimal numbers with optional single decimal place
    pattern = r"^(<think>.*?</think>\s*)?SCORE:\s*(10\.0|10|[0-9](?:\.[0-9])?)\s*$"
    return bool(re.match(pattern, completion, re.DOTALL | re.IGNORECASE))


def compute_gaussian_reward(predicted: float, target: float, sigma: float = 2.5) -> float:
    """
    Gaussian reward function for decimal scores 0.0-10.0.

    Args:
        predicted: Predicted decimal score (0.0-10.0)
        target: Target decimal score (0.0-10.0)
        sigma: Standard deviation (default 2.5)

    Returns:
        Reward following Gaussian distribution
    """
    distance = abs(predicted - target)
    return np.exp(-(distance**2) / (2 * sigma**2))


def format_compliance_reward_fn(
    prompts: List[str], completions: List[str], bins: List[float], **_
) -> List[float]:
    """
    Reward function for format compliance.

    Args:
        prompts: List of input prompts
        completions: List of model completions  
        bins: List of target bin numbers (0.0-10.0 decimal values)

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
    prompts: List[str], completions: List[str], bins: List[float], **_
) -> List[float]:
    """
    Reward function for similarity rating accuracy.

    Args:
        prompts: List of input prompts
        completions: List of model completions
        bins: List of target bin numbers (0.0-10.0 decimal values)

    Returns:
        List of rewards (0.0 to 1.0) based on accuracy of similarity rating
    """
    rewards = []
    for completion, target in zip(completions, bins):  # Fixed: was using undefined 'scores'
        predicted_rating = extract_rating_from_completion(completion)

        if predicted_rating == -1.0:
            rewards.append(0.0)
            continue

        reward = compute_gaussian_reward(predicted_rating, target)
        rewards.append(reward)

    return rewards
