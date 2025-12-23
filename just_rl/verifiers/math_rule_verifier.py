import re

class MathRuleVerifier:
    """
    A simple rule-based verifier for math problems.
    
    Logic:
    1. Extract the last occurrence of content within \\boxed{...}.
    2. Normalize the extracted content and the ground truth (strip whitespace, remove spaces).
    3. Compare for exact match.
    4. Return 1.0 if match, 0.0 otherwise.
    
    Constraints:
    - No SymPy.
    - No robust formatting tolerance.
    """
    
    def __init__(self):
        # Regex to capture content inside \boxed{...}
        # Note: This simple regex might fail on nested braces, but the requirements 
        # specify a "simple regex-based binary reward function" and specifically 
        # mention r'\\boxed\{(.*?)\}'.
        self.boxed_pattern = re.compile(r'\\boxed\{(.*?)\}')

    def extract_answer(self, text: str) -> str:
        """
        Extracts the content of the last \boxed{...} in the text.
        Returns None if no boxed content is found.
        """
        matches = self.boxed_pattern.findall(text)
        if not matches:
            return None
        return matches[-1]

    def normalize(self, text: str) -> str:
        """
        Normalizes text by stripping whitespace and removing all spaces.
        """
        if text is None:
            return ""
        # Strip whitespace from ends
        text = text.strip()
        # Remove all internal spaces
        text = text.replace(" ", "")
        return text

    def verify(self, generation: str, ground_truth: str) -> float:
        """
        Verifies the generation against the ground truth.
        
        Args:
            generation: The model's full output text.
            ground_truth: The expected answer string (usually the content inside boxed).
            
        Returns:
            1.0 if the extracted answer matches the ground truth exactly (after normalization).
            0.0 otherwise.
        """
        extracted = self.extract_answer(generation)
        
        if extracted is None:
            return 0.0
            
        norm_extracted = self.normalize(extracted)
        norm_gt = self.normalize(ground_truth)
        
        if norm_extracted == norm_gt:
            return 1.0
        
        return 0.0

# Example usage for testing
if __name__ == "__main__":
    verifier = MathRuleVerifier()
    
    # Test cases
    test_gen_1 = "The answer is \\boxed{42}."
    gt_1 = "42"
    print(f"Test 1 (Match): {verifier.verify(test_gen_1, gt_1)}") # Should be 1.0
    
    test_gen_2 = "The answer is \\boxed{ 42 }."
    gt_2 = "42"
    print(f"Test 2 (Space Match): {verifier.verify(test_gen_2, gt_2)}") # Should be 1.0
    
    test_gen_3 = "The answer is \\boxed{43}."
    gt_3 = "42"
    print(f"Test 3 (Mismatch): {verifier.verify(test_gen_3, gt_3)}") # Should be 0.0
    
    test_gen_4 = "No boxed answer here."
    gt_4 = "42"
    print(f"Test 4 (No Box): {verifier.verify(test_gen_4, gt_4)}") # Should be 0.0
