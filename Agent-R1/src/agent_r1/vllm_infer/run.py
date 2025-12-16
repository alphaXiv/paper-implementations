#!/usr/bin/env python3
"""
Script to run inference with configurable parameters
"""

import argparse
import json
import importlib
import os
from openai import OpenAI

from agent_r1.tool.envs.nous import NousToolEnv
from agent_r1.tool.tools import _default_tool
import agent_r1.vllm_infer.config as default_config

# ANSI color codes for colored output
COLORS = {
    "user": "\033[1;34m",      # Bold Blue
    "assistant": "\033[1;32m",  # Bold Green
    "tool": "\033[1;33m",       # Bold Yellow
    "tool_call": "\033[1;35m",  # Bold Purple
    "reset": "\033[0m",         # Reset to default
    "bg_user": "\033[44m",      # Blue background
    "bg_assistant": "\033[42m", # Green background
    "bg_tool": "\033[43m",      # Yellow background
    "bg_tool_call": "\033[45m", # Purple background
}

def parse_args():
    parser = argparse.ArgumentParser(description='Run VLLM inference with configurable parameters')
    
    # Environment and API settings
    parser.add_argument('--tools', type=str, default=default_config.TOOLS,
                        help='Environment for tool selection')
    parser.add_argument('--api-key', type=str, default=default_config.OPENAI_API_KEY,
                        help='OpenAI API key')
    parser.add_argument('--api-base', type=str, default=default_config.OPENAI_API_BASE,
                        help='OpenAI API base URL')
    parser.add_argument('--model', type=str, default=default_config.MODEL_NAME,
                        help='Model name for inference')
    
    # Model inference parameters
    parser.add_argument('--temperature', type=float, default=default_config.TEMPERATURE,
                        help='Temperature for sampling')
    parser.add_argument('--top-p', type=float, default=default_config.TOP_P,
                        help='Top-p for nucleus sampling')
    parser.add_argument('--max-tokens', type=int, default=default_config.MAX_TOKENS,
                        help='Maximum number of tokens to generate')
    parser.add_argument('--repetition-penalty', type=float, default=default_config.REPETITION_PENALTY,
                        help='Repetition penalty for generation')
    
    # Question parameter
    parser.add_argument('--question', type=str, default="Who is older, James Harden or Stephen Curry?",
                        help='Question to ask the model')
    
    # Config file
    parser.add_argument('--config', type=str, default=None,
                        help='Path to custom config file to override defaults')
    
    # Add option to disable colors
    parser.add_argument('--no-color', action='store_true',
                        help='Disable colored output')
    
    return parser.parse_args()

def load_custom_config(config_path):
    """Load custom configuration from a Python file"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    spec = importlib.util.spec_from_file_location("custom_config", config_path)
    custom_config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(custom_config)
    return custom_config

def main():
    args = parse_args()
    
    # Check if colors should be disabled
    use_colors = not args.no_color
    
    # Load custom config if provided
    config = default_config
    if args.config:
        try:
            config = load_custom_config(args.config)
            print(f"Loaded custom config from {args.config}")
        except Exception as e:
            print(f"Error loading custom config: {e}")
            print("Falling back to default config")
    
    # Override config with command-line arguments
    TOOLS = args.tools
    OPENAI_API_KEY = args.api_key
    OPENAI_API_BASE = args.api_base
    MODEL_NAME = args.model
    TEMPERATURE = args.temperature
    TOP_P = args.top_p
    MAX_TOKENS = args.max_tokens
    REPETITION_PENALTY = args.repetition_penalty
    INSTRUCTION_FOLLOWING = config.INSTRUCTION_FOLLOWING
    
    # Initialize OpenAI client
    client = OpenAI(
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_API_BASE,
    )
    
    # Set up tools
    tools = []
    for tool in TOOLS:
        tools.append(_default_tool(tool))
    env = NousToolEnv(tools=tools, max_tool_response_length=MAX_TOKENS)
    
    # Create message with question
    question_raw = args.question
    messages = [{
        "role": "user",
        "content": "Question: " + question_raw + '\n' + INSTRUCTION_FOLLOWING
    }]
    
    print(f"Running inference with model: {MODEL_NAME}")
    if use_colors:
        print(f"{COLORS['bg_user']} User {COLORS['reset']} {COLORS['user']}{question_raw}{COLORS['reset']}")
    else:
        print(f"User: {question_raw}")

    tools = [tool.tool_description for tool in tools]
    
    # Run inference loop
    while True:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            tools=tools,
            tool_choice="auto",
            temperature=TEMPERATURE,
            top_p=TOP_P,
            max_tokens=MAX_TOKENS,
            extra_body={
                "repetition_penalty": REPETITION_PENALTY,
            },
            stop=["</tool_call>"]
        )
        
        # Get the response message
        response_message = response.choices[0].message
        
        # Format the assistant's message properly
        assistant_message = {
            "role": "assistant",
            "content": response_message.content
        }
        
        # Add tool calls if any
        if response_message.tool_calls:
            assistant_message["tool_calls"] = [
                {
                    "id": tool_call.id,
                    "type": tool_call.type,
                    "function": {
                        "name": tool_call.function.name,
                        "arguments": tool_call.function.arguments
                    }
                }
                for tool_call in response_message.tool_calls
            ]
        
        # Add the formatted message to the conversation
        messages.append(assistant_message)
        
        # Display assistant's response with color
        if use_colors:
            print(f"\n{COLORS['bg_assistant']} Assistant {COLORS['reset']} {COLORS['assistant']}{response_message.content}{COLORS['reset']}")
        else:
            print(f"\nAssistant: {response_message.content}")
        
        # Check if there are any tool calls
        if response_message.tool_calls:
            # Process each tool call
            for tool_call in response_message.tool_calls:
                # Pretty format the arguments for better readability
                try:
                    args_dict = json.loads(tool_call.function.arguments)
                    formatted_args = json.dumps(args_dict, indent=2)
                except json.JSONDecodeError:
                    formatted_args = tool_call.function.arguments
                
                # Log function call details with color
                if use_colors:
                    print(f"\n{COLORS['bg_tool_call']} Tool Call {COLORS['reset']} {COLORS['tool_call']}Function: {tool_call.function.name}{COLORS['reset']}")
                    print(f"{COLORS['tool_call']}Arguments:{COLORS['reset']}\n{formatted_args}")
                else:
                    print(f"\n[Tool Call] Function: {tool_call.function.name}")
                    print(f"Arguments:\n{formatted_args}")
                
                # Execute the tool
                result = env.tool_map[tool_call.function.name].execute(json.loads(tool_call.function.arguments))
                result = result["content"]
                
                # Display tool result with color
                if use_colors:
                    print(f"\n{COLORS['bg_tool']} Tool {COLORS['reset']} {COLORS['tool']}{result}{COLORS['reset']}")
                else:
                    print(f"\nTool: {result}")
                
                # Add tool result to messages
                messages.append({
                    "role": "tool",
                    "content": result,
                    "tool_call_id": tool_call.id
                })
        else:
            # No tool calls, we have reached the final answer
            break

if __name__ == "__main__":
    main() 