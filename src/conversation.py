import requests
import json
import os

class ConversationManager:
    """
    Manages multi-turn, history-aware conversations with an LLM (e.g., Claude).
    Stores conversation history and provides methods for sending prompts and follow-ups.
    Designed for CLI or frontend integration.
    """
    DEFAULT_SYSTEM_PROMPT = (
        "You are an expert in deep learning model training. "
        "You will be given experiment summaries and user questions. "
        "Provide concise, actionable advice for improving model performance."
    )

    def __init__(self, api_url=None, api_key=None, model="claude-3-opus-20240229", max_tokens=512, system_prompt=None):
        self.api_url = api_url or os.environ.get("CLAUDE_API_URL", "https://api.anthropic.com/v1/messages")
        self.api_key = api_key or os.environ.get("CLAUDE_API_KEY")
        self.model = model
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT
        self.history = []  # List of {role, content}

    def _get_headers(self):
        return {
            "x-api-key": self.api_key,
            "content-type": "application/json",
            "anthropic-version": "2023-06-01"
        }

    def send_message(self, prompt, role="user"):
        """
        Sends a message to the LLM and appends to conversation history.
        Returns the LLM's response text.
        """
        if not self.api_key:
            print("No Claude API key found in CLAUDE_API_KEY environment variable.")
            return None
        # Anthropic API expects system prompt as a top-level param, not a message
        data = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": self.system_prompt,
            "messages": self.history + [{"role": role, "content": prompt}]
        }
        self.history.append({"role": role, "content": prompt})
        try:
            response = requests.post(self.api_url, headers=self._get_headers(), data=json.dumps(data))
            if response.status_code == 200:
                result = response.json()
                # Claude's response format: result['content'] is a list of dicts with 'text'
                if "content" in result and isinstance(result["content"], list):
                    text = result["content"][0].get("text", result["content"][0])
                else:
                    text = str(result)
                self.history.append({"role": "assistant", "content": text})
                return text
            else:
                print("Error calling Claude API:", response.text)
                return None
        except Exception as e:
            print("Exception during Claude API call:", e)
            return None

    def ask_followup(self, user_question, system_prompt=None):
        """
        Handles a follow-up user question, optionally with a system prompt for context.
        Returns the LLM's response text.
        """
        if not user_question.strip():
            return None
        followup_prompt = user_question
        if system_prompt:
            followup_prompt = f"{system_prompt}\n{user_question}"
        return self.send_message(followup_prompt, role="user")

    def get_history(self):
        return self.history

    def clear_history(self):
        self.history = []
