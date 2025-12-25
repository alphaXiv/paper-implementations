"""
Configuration parameters for the VLLM inference
"""

# Environment and API settings
TOOLS = ["search"]
OPENAI_API_KEY = "EMPTY"
OPENAI_API_BASE = "http://localhost:8000/v1"
MODEL_NAME = "agent"

# Model inference parameters
TEMPERATURE = 0.7
TOP_P = 0.8
MAX_TOKENS = 512
REPETITION_PENALTY = 1.05

INSTRUCTION_FOLLOWING = (
    r'You FIRST think about the reasoning process as an internal monologue and then provide the final answer. '
    r'The reasoning process MUST BE enclosed within <think> </think> tags. The final answer MUST BE put in <answer> </answer> tags.'
)