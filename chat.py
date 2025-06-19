#!/usr/bin/env python3
"""
Conversational AI Assistant using Amazon Bedrock with Llama 4 Maverick 17B Instruct

This module provides a complete conversational AI system with context management,
error handling, and comprehensive testing capabilities.

Author: Assistant
Date: 2025
"""

import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import uuid
import os

import boto3
from botocore.exceptions import ClientError, BotoCoreError
from botocore.config import Config


# Configuration and Data Models
@dataclass
class ConversationContext:
    """Manages conversation context and memory."""
    conversation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    messages: List[Dict[str, str]] = field(default_factory=list)
    user_preferences: Dict[str, Any] = field(default_factory=dict)
    session_start: datetime = field(default_factory=datetime.now)
    last_interaction: datetime = field(default_factory=datetime.now)
    max_context_length: int = 10
    
    def add_message(self, role: str, content: str) -> None:
        """Add a message to conversation history."""
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        }
        self.messages.append(message)
        self.last_interaction = datetime.now()
        
        # Trim context if too long
        if len(self.messages) > self.max_context_length * 2:  # *2 for user+assistant pairs
            self.messages = self.messages[-self.max_context_length * 2:]
    
    def get_context_string(self) -> str:
        """Get formatted context for the AI model."""
        if not self.messages:
            return ""
        
        context_parts = []
        for msg in self.messages[-self.max_context_length:]:
            context_parts.append(f"{msg['role']}: {msg['content']}")
        
        return "\n".join(context_parts)
    
    def is_expired(self, timeout_minutes: int = 30) -> bool:
        """Check if conversation context has expired."""
        return datetime.now() - self.last_interaction > timedelta(minutes=timeout_minutes)


@dataclass
class BedrockConfig:
    """Configuration for Amazon Bedrock."""
    model_id: str = "us.meta.llama4-maverick-17b-instruct-v1:0"  # Using inference profile
    region_name: str = "us-east-1"
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    retry_attempts: int = 3
    timeout_seconds: int = 30


class MemoryStore(ABC):
    """Abstract base class for conversation memory storage."""
    
    @abstractmethod
    def save_context(self, context: ConversationContext) -> bool:
        """Save conversation context."""
        pass
    
    @abstractmethod
    def load_context(self, conversation_id: str) -> Optional[ConversationContext]:
        """Load conversation context."""
        pass
    
    @abstractmethod
    def cleanup_expired(self, timeout_minutes: int = 30) -> int:
        """Clean up expired conversations."""
        pass


class InMemoryStore(MemoryStore):
    """In-memory implementation of conversation storage."""
    
    def __init__(self):
        self._store: Dict[str, ConversationContext] = {}
        self.logger = logging.getLogger(__name__)
    
    def save_context(self, context: ConversationContext) -> bool:
        """Save conversation context to memory."""
        try:
            self._store[context.conversation_id] = context
            self.logger.debug(f"Saved context for conversation {context.conversation_id}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to save context: {e}")
            return False
    
    def load_context(self, conversation_id: str) -> Optional[ConversationContext]:
        """Load conversation context from memory."""
        context = self._store.get(conversation_id)
        if context and context.is_expired():
            del self._store[conversation_id]
            return None
        return context
    
    def cleanup_expired(self, timeout_minutes: int = 30) -> int:
        """Clean up expired conversations."""
        expired_ids = [
            cid for cid, ctx in self._store.items()
            if ctx.is_expired(timeout_minutes)
        ]
        
        for cid in expired_ids:
            del self._store[cid]
        
        self.logger.info(f"Cleaned up {len(expired_ids)} expired conversations")
        return len(expired_ids)


class BedrockAIError(Exception):
    """Custom exception for Bedrock AI operations."""
    pass


class ConversationalAI:
    """Main conversational AI assistant using Amazon Bedrock."""
    
    def __init__(self, config: Optional[BedrockConfig] = None, 
                 memory_store: Optional[MemoryStore] = None):
        """Initialize the conversational AI assistant."""
        self.config = config or BedrockConfig()
        self.memory_store = memory_store or InMemoryStore()
        self.logger = self._setup_logging()
        
        # Initialize Bedrock client with best practices
        self.bedrock_client = self._create_bedrock_client()
        
        # Test connection
        self._test_connection()
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration."""
        logger = logging.getLogger(__name__)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        logger.setLevel(logging.INFO)
        return logger
    
    def _create_bedrock_client(self):
        """Create Bedrock client with best practices."""
        try:
            # Configuration for retry and timeout
            config = Config(
                retries={
                    'max_attempts': self.config.retry_attempts,
                    'mode': 'adaptive'
                },
                read_timeout=self.config.timeout_seconds,
                connect_timeout=10
            )
            
            # Create session and client
            session = boto3.Session()
            client = session.client(
                'bedrock-runtime',
                region_name=self.config.region_name,
                config=config
            )
            
            self.logger.info(f"Bedrock client initialized for region {self.config.region_name}")
            return client
            
        except Exception as e:
            self.logger.error(f"Failed to create Bedrock client: {e}")
            raise BedrockAIError(f"Client initialization failed: {e}")
    
    def _test_connection(self) -> None:
        """Test Bedrock connection."""
        try:
            # Simple test call
            self._invoke_model("Hello", max_tokens=10)
            self.logger.info("Bedrock connection test successful")
        except Exception as e:
            self.logger.warning(f"Bedrock connection test failed: {e}")
    
    def _invoke_model(self, prompt: str, max_tokens: Optional[int] = None) -> str:
        """Invoke the Bedrock model with error handling and retries."""
        max_tokens = max_tokens or self.config.max_tokens
        
        # Prepare request body for Llama model
        request_body = {
            "prompt": prompt,
            "max_gen_len": max_tokens,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p
        }
        
        for attempt in range(self.config.retry_attempts):
            try:
                self.logger.debug(f"Invoking model (attempt {attempt + 1}/{self.config.retry_attempts})")
                response = self.bedrock_client.invoke_model(
                    modelId=self.config.model_id,
                    body=json.dumps(request_body),
                    contentType='application/json'
                )
                # Parse response
                response_body = json.loads(response['body'].read())
                if 'generation' in response_body:
                    return response_body['generation'].strip()
                else:
                    raise BedrockAIError("Unexpected response format")
            except ClientError as e:
                error_code = e.response['Error']['Code']
                error_message = e.response['Error']['Message']
                self.logger.error(f"AWS API error (attempt {attempt + 1}): {error_code} - {error_message}")
                if error_code in ['ThrottlingException', 'ServiceUnavailableException']:
                    if attempt < self.config.retry_attempts - 1:
                        time.sleep(2 ** attempt)  # Exponential backoff
                        continue
                raise BedrockAIError(f"AWS API error: {error_code} - {error_message}")
            except BotoCoreError as e:
                self.logger.error(f"Boto3 error (attempt {attempt + 1}): {e}")
                if attempt < self.config.retry_attempts - 1:
                    time.sleep(2 ** attempt)
                    continue
                raise BedrockAIError(f"Connection error: {e}")
            except StopIteration as e:
                self.logger.error(f"StopIteration in mock (attempt {attempt + 1}): {e}")
                raise BedrockAIError("Mock ran out of responses during retry attempts")
            except Exception as e:
                self.logger.error(f"Unexpected error (attempt {attempt + 1}): {e}")
                if attempt < self.config.retry_attempts - 1:
                    time.sleep(1)
                    continue
                raise BedrockAIError(f"Unexpected error: {e}")
        raise BedrockAIError("Max retry attempts exceeded")
    
    def chat(self, message: str, conversation_id: Optional[str] = None) -> Dict[str, Any]:
        """Process a chat message and return response with context."""
        try:
            # Load or create conversation context
            if conversation_id:
                context = self.memory_store.load_context(conversation_id)
                if not context:
                    self.logger.info(f"Creating new context for ID {conversation_id}")
                    context = ConversationContext(conversation_id=conversation_id)
            else:
                context = ConversationContext()
                conversation_id = context.conversation_id
            
            # Add user message to context
            context.add_message("user", message)
            
            # Build prompt with context
            prompt = self._build_prompt(message, context)
            
            # Get AI response
            ai_response = self._invoke_model(prompt)
            
            # Add AI response to context
            context.add_message("assistant", ai_response)
            
            # Save context
            self.memory_store.save_context(context)
            
            # Cleanup expired conversations periodically
            if len(context.messages) % 10 == 0:  # Every 10 messages
                self.memory_store.cleanup_expired()
            
            self.logger.info(f"Processed message for conversation {conversation_id}")
            
            return {
                "response": ai_response,
                "conversation_id": conversation_id,
                "message_count": len(context.messages),
                "timestamp": datetime.now().isoformat()
            }
            
        except BedrockAIError:
            raise  # Re-raise Bedrock-specific errors
        except Exception as e:
            self.logger.error(f"Chat processing error: {e}")
            raise BedrockAIError(f"Chat processing failed: {e}")
    
    def _build_prompt(self, message: str, context: ConversationContext) -> str:
        """Build prompt with conversation context."""
        system_prompt = (
            "You are a helpful AI assistant. Provide clear, concise, and helpful responses. "
            "Use the conversation history to maintain context and provide relevant answers."
        )
        
        if context.messages:
            conversation_history = context.get_context_string()
            prompt = f"{system_prompt}\n\nConversation history:\n{conversation_history}\n\nuser: {message}\nassistant:"
        else:
            prompt = f"{system_prompt}\n\nuser: {message}\nassistant:"
        
        return prompt
    
    def get_conversation_info(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a conversation."""
        context = self.memory_store.load_context(conversation_id)
        if not context:
            return None
        
        return {
            "conversation_id": context.conversation_id,
            "message_count": len(context.messages),
            "session_start": context.session_start.isoformat(),
            "last_interaction": context.last_interaction.isoformat(),
            "user_preferences": context.user_preferences
        }
    
    def reset_conversation(self, conversation_id: str) -> bool:
        """Reset a conversation context."""
        context = self.memory_store.load_context(conversation_id)
        if context:
            context.messages.clear()
            context.last_interaction = datetime.now()
            return self.memory_store.save_context(context)
        return False


# Example usage and CLI interface
def main():
    """Main function for CLI usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Conversational AI Assistant")
    parser.add_argument("--region", default="us-east-1", help="AWS region")
    parser.add_argument("--model", default="us.meta.llama4-maverick-17b-instruct-v1:0", help="Model ID or Inference Profile ID")
    parser.add_argument("--max-tokens", type=int, default=512, help="Max tokens")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature")
    parser.add_argument("--conversation-id", help="Conversation ID to continue")
    
    args = parser.parse_args()
    
    # Create configuration
    config = BedrockConfig(
        model_id=args.model,
        region_name=args.region,
        max_tokens=args.max_tokens,
        temperature=args.temperature
    )
    
    # Initialize AI assistant
    try:
        ai = ConversationalAI(config)
        print("🤖 Conversational AI Assistant Ready!")
        print("Type 'quit' to exit, 'info' for conversation details, 'reset' to clear history")
        print("-" * 60)
        
        conversation_id = args.conversation_id
        
        while True:
            try:
                user_input = input("\n👤 You: ").strip()
                
                if user_input.lower() == 'quit':
                    print("👋 Goodbye!")
                    break
                elif user_input.lower() == 'info':
                    if conversation_id:
                        info = ai.get_conversation_info(conversation_id)
                        if info:
                            print(f"📊 Conversation Info: {json.dumps(info, indent=2)}")
                        else:
                            print("❌ No conversation found")
                    else:
                        print("❌ No active conversation")
                    continue
                elif user_input.lower() == 'reset':
                    if conversation_id and ai.reset_conversation(conversation_id):
                        print("🔄 Conversation reset")
                    else:
                        print("❌ Reset failed")
                    continue
                elif not user_input:
                    continue
                
                # Get AI response
                result = ai.chat(user_input, conversation_id)
                conversation_id = result["conversation_id"]
                
                print(f"\n🤖 Assistant: {result['response']}")
                print(f"💬 Messages: {result['message_count']} | ID: {conversation_id[:8]}...")
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                
    except Exception as e:
        print(f"❌ Failed to initialize AI assistant: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
