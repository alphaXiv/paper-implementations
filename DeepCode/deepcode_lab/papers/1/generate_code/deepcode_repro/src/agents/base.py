import os
import yaml
import json
import time
from typing import List, Dict, Any, Optional, Union
from abc import ABC, abstractmethod
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

import openai
import anthropic
from pydantic import BaseModel

from ..utils.logger import logger
from ..utils.mcp_tools import MCPToolkit

class AgentConfig(BaseModel):
    model_provider: str  # "anthropic" or "openai"
    model_name: str
    temperature: float = 0.0
    max_tokens: int = 4096
    api_key: Optional[str] = None

class BaseAgent(ABC):
    """
    Base class for all DeepCode agents.
    Handles LLM interaction, tool execution, and retry logic.
    """

    def __init__(self, name: str, config: Dict[str, Any], mcp_toolkit: Optional[MCPToolkit] = None):
        """
        Initialize the agent.
        
        Args:
            name: Name of the agent (for logging).
            config: Configuration dictionary (usually from config.yaml).
            mcp_toolkit: Optional toolkit for tool execution.
        """
        self.name = name
        self.config = config
        self.mcp_toolkit = mcp_toolkit
        
        # Load LLM settings
        llm_cfg = config.get("llm", {})
        self.provider = llm_cfg.get("provider", "anthropic")
        self.model_name = llm_cfg.get("model", "claude-3-5-sonnet-20240620")
        self.temperature = llm_cfg.get("temperature", 0.0)
        self.max_tokens = llm_cfg.get("max_tokens", 4096)
        
        # Initialize clients
        self._init_client()
        
        logger.info(f"Agent '{self.name}' initialized with {self.provider}/{self.model_name}")

    def _init_client(self):
        """Initialize the appropriate LLM client based on configuration."""
        if self.provider == "anthropic":
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                logger.warning(f"ANTHROPIC_API_KEY not found in environment for agent {self.name}")
            self.client = anthropic.Anthropic(api_key=api_key)
        elif self.provider == "openai":
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                logger.warning(f"OPENAI_API_KEY not found in environment for agent {self.name}")
            self.client = openai.OpenAI(api_key=api_key)
        else:
            raise ValueError(f"Unsupported model provider: {self.provider}")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((
            anthropic.APIConnectionError, 
            anthropic.RateLimitError,
            openai.APIConnectionError,
            openai.RateLimitError
        ))
    )
    def call_llm(self, 
                 system_prompt: str, 
                 user_message: str, 
                 tools: Optional[List[Dict]] = None) -> str:
        """
        Execute a call to the LLM with retry logic.
        
        Args:
            system_prompt: The system instruction.
            user_message: The user query or context.
            tools: Optional list of tool definitions (if supported).
            
        Returns:
            The text response from the LLM.
        """
        logger.debug(f"Agent '{self.name}' calling LLM...")
        
        try:
            if self.provider == "anthropic":
                return self._call_anthropic(system_prompt, user_message, tools)
            elif self.provider == "openai":
                return self._call_openai(system_prompt, user_message, tools)
            else:
                raise ValueError(f"Unknown provider {self.provider}")
        except Exception as e:
            logger.error(f"Error calling LLM in agent '{self.name}': {str(e)}")
            raise

    def _call_anthropic(self, system_prompt: str, user_message: str, tools: Optional[List[Dict]] = None) -> str:
        """Handle Anthropic API call."""
        messages = [{"role": "user", "content": user_message}]
        
        # Note: Anthropic tool use is slightly different, but for this reproduction 
        # we might stick to text-based tool use or standard API if needed.
        # For simplicity in this base class, we'll focus on text generation.
        # If tools are provided, we would format them into the system prompt or use the tool API.
        
        # If tools are strictly required via API:
        kwargs = {
            "model": self.model_name,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "system": system_prompt,
            "messages": messages
        }
        
        # Basic tool integration if needed (Anthropic beta tools)
        # For now, we assume the prompt handles tool instructions or we use the official tool param if available
        # In this repro, we will rely on the prompt to guide the model to use tools if we aren't using the native tool calling API fully yet,
        # OR we can implement native tool calling if the library version supports it.
        # Given requirements.txt has anthropic>=0.7.0, it supports messages API.
        
        response = self.client.messages.create(**kwargs)
        
        # Extract text content
        if response.content and len(response.content) > 0:
            return response.content[0].text
        return ""

    def _call_openai(self, system_prompt: str, user_message: str, tools: Optional[List[Dict]] = None) -> str:
        """Handle OpenAI API call."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]
        
        kwargs = {
            "model": self.model_name,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }
        
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
            
        response = self.client.chat.completions.create(**kwargs)
        return response.choices[0].message.content

    def run_tool(self, tool_name: str, **kwargs) -> Any:
        """
        Execute a tool via the MCP Toolkit.
        
        Args:
            tool_name: Name of the tool to run (e.g., 'read_file', 'execute_command').
            **kwargs: Arguments for the tool.
            
        Returns:
            Tool execution result.
        """
        if not self.mcp_toolkit:
            raise ValueError("MCP Toolkit not initialized for this agent.")
            
        logger.info(f"Agent '{self.name}' executing tool: {tool_name}")
        
        # Map string names to toolkit methods
        # This is a simple dispatcher. In a full MCP implementation, this might be more dynamic.
        if tool_name == "read_file":
            return self.mcp_toolkit.fs.read_file(kwargs.get("path"))
        elif tool_name == "write_file":
            return self.mcp_toolkit.fs.write_file(kwargs.get("path"), kwargs.get("content"))
        elif tool_name == "list_files":
            return self.mcp_toolkit.fs.list_files(kwargs.get("path", "."))
        elif tool_name == "execute_command":
            return self.mcp_toolkit.sandbox.execute_command(kwargs.get("command"), kwargs.get("timeout", 30))
        elif tool_name == "search":
            return self.mcp_toolkit.search.search(kwargs.get("query"), kwargs.get("count", 5))
        else:
            return f"Error: Tool '{tool_name}' not found."

    @abstractmethod
    def run(self, *args, **kwargs) -> Any:
        """Main execution method for the agent."""
        pass
