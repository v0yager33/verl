"""
Multiple choice reward function for virl39k and spatial_dise datasets.
"""

import re


def compute_score(solution_str: str, ground_truth: str) -> float:
    """
    Compute score for multiple choice questions.
    
    Args:
        solution_str: The model's solution/response string
        ground_truth: The ground truth answer (e.g., "A", "B", "C", "D")
    
    Returns:
        float: 1.0 if correct, 0.0 otherwise
    """
    # Extract the answer from solution_str
    # Try multiple patterns to extract the answer
    
    # Pattern 1: \boxed{X}
    match = re.search(r'\\boxed\{([A-Da-d])\}', solution_str)
    if match:
        predicted = match.group(1).upper()
        return float(predicted == ground_truth.upper())
    
    # Pattern 2: \boxed{X} (without escape)
    match = re.search(r'boxed\{([A-Da-d])\}', solution_str)
    if match:
        predicted = match.group(1).upper()
        return float(predicted == ground_truth.upper())
    
    # Pattern 3: Answer is X or The answer is X
    match = re.search(r'(?:answer|choice|option)\s*(?:is)?\s*:?\s*([A-Da-d])', solution_str, re.IGNORECASE)
    if match:
        predicted = match.group(1).upper()
        return float(predicted == ground_truth.upper())
    
    # Pattern 4: Standalone letter at the end or after certain keywords
    match = re.search(r'(?:^|\s|\.|,)([A-Da-d])(?:\s*$|\s*\.|\s*,)', solution_str)
    if match:
        predicted = match.group(1).upper()
        return float(predicted == ground_truth.upper())
    
    # Pattern 5: Any single letter A-D in the response
    matches = re.findall(r'\b([A-Da-d])\b', solution_str)
    if matches:
        # Take the last one as the answer (usually the final answer)
        predicted = matches[-1].upper()
        return float(predicted == ground_truth.upper())
    
    # No valid answer found
    return 0.0
