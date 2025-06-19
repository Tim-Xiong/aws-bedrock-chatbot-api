#!/usr/bin/env python3
"""
Test suite for the Conversational AI Assistant.

This module provides comprehensive testing for all components of the
conversational AI system with mocking for AWS services.
"""

import json
import pytest
import uuid
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
from botocore.exceptions import ClientError, BotoCoreError

# Import the modules to test
from chat import (
    ConversationContext,
    BedrockConfig,
    InMemoryStore,
    ConversationalAI,
    BedrockAIError,
    main
)


class TestConversationContext:
    """Test cases for ConversationContext class."""

    def test_initialization(self):
        """Test ConversationContext initialization."""
        context = ConversationContext()
        
        assert isinstance(context.conversation_id, str)
        assert len(context.messages) == 0
        assert isinstance(context.user_preferences, dict)
        assert isinstance(context.session_start, datetime)
        assert context.max_context_length == 10

    def test_add_message(self):
        """Test adding messages to conversation context."""
        context = ConversationContext()
        
        context.add_message("user", "Hello")
        assert len(context.messages) == 1
        assert context.messages[0]["role"] == "user"
        assert context.messages[0]["content"] == "Hello"
        assert "timestamp" in context.messages[0]

    def test_context_trimming(self):
        """Test context trimming when max length exceeded."""
        context = ConversationContext(max_context_length=2)
        
        # Add messages beyond limit
        for i in range(6):  # 6 > 2*2 (max_context_length * 2)
            context.add_message("user", f"Message {i}")
        
        assert len(context.messages) == 4  # Should keep last 4 messages

    def test_get_context_string(self):
        """Test getting formatted context string."""
        context = ConversationContext()
        
        # Empty context
        assert context.get_context_string() == ""
        
        # With messages
        context.add_message("user", "Hello")
        context.add_message("assistant", "Hi there!")
        
        context_str = context.get_context_string()
        assert "user: Hello" in context_str
        assert "assistant: Hi there!" in context_str

    def test_is_expired(self):
        """Test conversation expiration check."""
        context = ConversationContext()
        
        # Fresh context should not be expired
        assert not context.is_expired(30)
        
        # Manually set old timestamp
        context.last_interaction = datetime.now() - timedelta(minutes=31)
        assert context.is_expired(30)


class TestBedrockConfig:
    """Test cases for BedrockConfig class."""

    def test_default_initialization(self):
        """Test BedrockConfig with default values."""
        config = BedrockConfig()
        
        assert config.model_id == "us.meta.llama4-maverick-17b-instruct-v1:0"
        assert config.region_name == "us-east-1"
        assert config.max_tokens == 512
        assert config.temperature == 0.7
        assert config.top_p == 0.9
        assert config.retry_attempts == 2
        assert config.timeout_seconds == 30

    def test_custom_initialization(self):
        """Test BedrockConfig with custom values."""
        config = BedrockConfig(
            model_id="custom-model",
            region_name="us-west-2",
            max_tokens=1024,
            temperature=0.5
        )
        
        assert config.model_id == "custom-model"
        assert config.region_name == "us-west-2"
        assert config.max_tokens == 1024
        assert config.temperature == 0.5


class TestInMemoryStore:
    """Test cases for InMemoryStore class."""

    def test_save_and_load_context(self):
        """Test saving and loading conversation context."""
        store = InMemoryStore()
        context = ConversationContext()
        context.add_message("user", "Test message")
        
        # Save context
        assert store.save_context(context) is True
        
        # Load context
        loaded_context = store.load_context(context.conversation_id)
        assert loaded_context is not None
        assert loaded_context.conversation_id == context.conversation_id
        assert len(loaded_context.messages) == 1

    def test_load_nonexistent_context(self):
        """Test loading a context that doesn't exist."""
        store = InMemoryStore()
        
        result = store.load_context("nonexistent-id")
        assert result is None

    def test_load_expired_context(self):
        """Test loading an expired context."""
        store = InMemoryStore()
        context = ConversationContext()
        
        # Make context expired
        context.last_interaction = datetime.now() - timedelta(minutes=31)
        store.save_context(context)
        
        # Should return None for expired context
        result = store.load_context(context.conversation_id)
        assert result is None

    def test_cleanup_expired(self):
        """Test cleanup of expired conversations."""
        store = InMemoryStore()
        
        # Create fresh and expired contexts
        fresh_context = ConversationContext()
        expired_context = ConversationContext()
        expired_context.last_interaction = datetime.now() - timedelta(minutes=31)
        
        store.save_context(fresh_context)
        store.save_context(expired_context)
        
        # Cleanup should remove 1 expired context
        cleaned_count = store.cleanup_expired(30)
        assert cleaned_count == 1
        
        # Fresh context should still exist
        assert store.load_context(fresh_context.conversation_id) is not None

    def test_save_context_error_handling(self):
        """Test error handling in save_context."""
        store = InMemoryStore()
        
        # Test with invalid context (None)
        result = store.save_context(None)
        assert result is False


class TestConversationalAI:
    """Test cases for ConversationalAI class."""

    @patch('chat.boto3.Session')
    def test_initialization(self, mock_session):
        """Test ConversationalAI initialization."""
        mock_client = Mock()
        mock_session.return_value.client.return_value = mock_client
        
        # Mock successful test connection
        mock_client.invoke_model.return_value = {
            'body': Mock(read=lambda: json.dumps({'generation': 'test'}))
        }
        
        ai = ConversationalAI()
        
        assert ai.config is not None
        assert ai.memory_store is not None
        assert ai.bedrock_client == mock_client

    @patch('chat.boto3.Session')
    def test_initialization_with_custom_config(self, mock_session):
        """Test initialization with custom config and memory store."""
        mock_client = Mock()
        mock_session.return_value.client.return_value = mock_client
        
        # Mock successful test connection
        mock_client.invoke_model.return_value = {
            'body': Mock(read=lambda: json.dumps({'generation': 'test'}))
        }
        
        config = BedrockConfig(model_id="custom-model")
        store = InMemoryStore()
        
        ai = ConversationalAI(config, store)
        
        assert ai.config.model_id == "custom-model"
        assert ai.memory_store == store

    @patch('chat.boto3.Session')
    def test_client_initialization_error(self, mock_session):
        """Test client initialization error handling."""
        mock_session.side_effect = Exception("AWS error")
        
        with pytest.raises(BedrockAIError):
            ConversationalAI()

    @patch('chat.boto3.Session')
    def test_invoke_model_success(self, mock_session):
        """Test successful model invocation."""
        mock_client = Mock()
        mock_session.return_value.client.return_value = mock_client
        
        # Mock successful response
        mock_response = {
            'body': Mock(read=lambda: json.dumps({'generation': 'Hello there!'}))
        }
        mock_client.invoke_model.return_value = mock_response
        
        ai = ConversationalAI()
        result = ai._invoke_model("Hello")
        
        assert result == "Hello there!"

    @patch('chat.boto3.Session')
    def test_invoke_model_client_error(self, mock_session):
        """Test model invocation with ClientError."""
        mock_client = Mock()
        mock_session.return_value.client.return_value = mock_client
        
        # Mock ClientError
        error_response = {
            'Error': {
                'Code': 'ValidationException',
                'Message': 'Invalid request'
            }
        }
        mock_client.invoke_model.side_effect = ClientError(error_response, 'invoke_model')
        
        ai = ConversationalAI()
        
        with pytest.raises(BedrockAIError):
            ai._invoke_model("Hello")

    @patch('chat.boto3.Session')
    def test_invoke_model_throttling_retry(self, mock_session):
        """Test model invocation with throttling and retry."""
        mock_client = Mock()
        mock_session.return_value.client.return_value = mock_client
        
        # Mock throttling error then success
        error_response = {
            'Error': {
                'Code': 'ThrottlingException',
                'Message': 'Rate exceeded'
            }
        }
        success_response = {
            'body': Mock(read=lambda: json.dumps({'generation': 'Success!'}))
        }
        
        mock_client.invoke_model.side_effect = [
            ClientError(error_response, 'invoke_model'),
            success_response
        ]
        
        ai = ConversationalAI()
        
        with patch('time.sleep'):  # Speed up test
            result = ai._invoke_model("Hello")
        
        assert result == "Success!"

    @patch('chat.boto3.Session')
    def test_chat_new_conversation(self, mock_session):
        """Test chat with new conversation."""
        mock_client = Mock()
        mock_session.return_value.client.return_value = mock_client
        
        mock_response = {
            'body': Mock(read=lambda: json.dumps({'generation': 'Nice to meet you!'}))
        }
        mock_client.invoke_model.return_value = mock_response
        
        ai = ConversationalAI()
        result = ai.chat("Hello!")
        
        assert result["response"] == "Nice to meet you!"
        assert "conversation_id" in result
        assert result["message_count"] == 2  # user + assistant
        assert "timestamp" in result

    @patch('chat.boto3.Session')
    def test_chat_existing_conversation(self, mock_session):
        """Test chat with existing conversation."""
        mock_client = Mock()
        mock_session.return_value.client.return_value = mock_client
        
        mock_response = {
            'body': Mock(read=lambda: json.dumps({'generation': 'How can I help?'}))
        }
        mock_client.invoke_model.return_value = mock_response
        
        ai = ConversationalAI()
        
        # First message
        result1 = ai.chat("Hello!")
        conversation_id = result1["conversation_id"]
        
        # Second message in same conversation
        result2 = ai.chat("How are you?", conversation_id)
        
        assert result2["conversation_id"] == conversation_id
        assert result2["message_count"] == 4  # 2 user + 2 assistant

    @patch('chat.boto3.Session')
    def test_chat_error_handling(self, mock_session):
        """Test chat error handling."""
        mock_client = Mock()
        mock_session.return_value.client.return_value = mock_client
        
        # Mock error
        mock_client.invoke_model.side_effect = Exception("Model error")
        
        ai = ConversationalAI()
        
        with pytest.raises(BedrockAIError):
            ai.chat("Hello!")

    @patch('chat.boto3.Session')
    def test_get_conversation_info(self, mock_session):
        """Test getting conversation information."""
        mock_client = Mock()
        mock_session.return_value.client.return_value = mock_client
        
        mock_response = {
            'body': Mock(read=lambda: json.dumps({'generation': 'Hello!'}))
        }
        mock_client.invoke_model.return_value = mock_response
        
        ai = ConversationalAI()
        
        # Create conversation
        result = ai.chat("Hello!")
        conversation_id = result["conversation_id"]
        
        # Get info
        info = ai.get_conversation_info(conversation_id)
        
        assert info is not None
        assert info["conversation_id"] == conversation_id
        assert info["message_count"] == 2
        assert "session_start" in info
        assert "last_interaction" in info

    @patch('chat.boto3.Session')
    def test_get_conversation_info_nonexistent(self, mock_session):
        """Test getting info for nonexistent conversation."""
        mock_client = Mock()
        mock_session.return_value.client.return_value = mock_client
        
        mock_response = {
            'body': Mock(read=lambda: json.dumps({'generation': 'Hello!'}))
        }
        mock_client.invoke_model.return_value = mock_response
        
        ai = ConversationalAI()
        
        info = ai.get_conversation_info("nonexistent-id")
        assert info is None

    @patch('chat.boto3.Session')
    def test_reset_conversation(self, mock_session):
        """Test resetting conversation."""
        mock_client = Mock()
        mock_session.return_value.client.return_value = mock_client
        
        mock_response = {
            'body': Mock(read=lambda: json.dumps({'generation': 'Hello!'}))
        }
        mock_client.invoke_model.return_value = mock_response
        
        ai = ConversationalAI()
        
        # Create conversation
        result = ai.chat("Hello!")
        conversation_id = result["conversation_id"]
        
        # Reset it
        reset_result = ai.reset_conversation(conversation_id)
        assert reset_result is True
        
        # Check messages are cleared
        info = ai.get_conversation_info(conversation_id)
        assert info["message_count"] == 0

    @patch('chat.boto3.Session')
    def test_reset_nonexistent_conversation(self, mock_session):
        """Test resetting nonexistent conversation."""
        mock_client = Mock()
        mock_session.return_value.client.return_value = mock_client
        
        mock_response = {
            'body': Mock(read=lambda: json.dumps({'generation': 'Hello!'}))
        }
        mock_client.invoke_model.return_value = mock_response
        
        ai = ConversationalAI()
        
        result = ai.reset_conversation("nonexistent-id")
        assert result is False

    @patch('chat.boto3.Session')
    def test_build_prompt_no_context(self, mock_session):
        """Test building prompt with no context."""
        mock_client = Mock()
        mock_session.return_value.client.return_value = mock_client
        
        mock_response = {
            'body': Mock(read=lambda: json.dumps({'generation': 'Hello!'}))
        }
        mock_client.invoke_model.return_value = mock_response
        
        ai = ConversationalAI()
        context = ConversationContext()
        
        prompt = ai._build_prompt("Hello!", context)
        
        assert "You are a helpful AI assistant" in prompt
        assert "user: Hello!" in prompt
        assert "assistant:" in prompt

    @patch('chat.boto3.Session')
    def test_build_prompt_with_context(self, mock_session):
        """Test building prompt with conversation context."""
        mock_client = Mock()
        mock_session.return_value.client.return_value = mock_client
        
        mock_response = {
            'body': Mock(read=lambda: json.dumps({'generation': 'Hello!'}))
        }
        mock_client.invoke_model.return_value = mock_response
        
        ai = ConversationalAI()
        context = ConversationContext()
        context.add_message("user", "Previous message")
        context.add_message("assistant", "Previous response")
        
        prompt = ai._build_prompt("New message", context)
        
        assert "Conversation history:" in prompt
        assert "Previous message" in prompt
        assert "Previous response" in prompt


class TestBedrockAIError:
    """Test cases for BedrockAIError exception."""

    def test_exception_creation(self):
        """Test creating BedrockAIError exception."""
        error = BedrockAIError("Test error message")
        assert str(error) == "Test error message"


class TestMainFunction:
    """Test cases for main CLI function."""

    @patch('chat.ConversationalAI')
    @patch('builtins.input')
    @patch('builtins.print')
    @patch('sys.argv', ['script.py'])
    def test_main_function_quit(self, mock_print, mock_input, mock_ai_class):
        """Test main function with quit command."""
        # Mock user input to quit immediately
        mock_input.return_value = "quit"
        
        # Mock AI instance
        mock_ai = Mock()
        mock_ai_class.return_value = mock_ai
        
        result = main()
        
        assert result == 0

    @patch('chat.ConversationalAI')
    @patch('builtins.input')
    @patch('builtins.print')
    @patch('sys.argv', ['script.py'])
    def test_main_function_chat(self, mock_print, mock_input, mock_ai_class):
        """Test main function with chat interaction."""
        # Mock user inputs: one message then quit
        mock_input.side_effect = ["Hello!", "quit"]
        
        # Mock AI instance and response
        mock_ai = Mock()
        mock_ai.chat.return_value = {
            "response": "Hi there!",
            "conversation_id": "test-id",
            "message_count": 2,
            "timestamp": "2025-01-01T00:00:00"
        }
        mock_ai_class.return_value = mock_ai
        
        result = main()
        
        assert result == 0
        mock_ai.chat.assert_called_once_with("Hello!", None)

    @patch('sys.argv', ['script.py'])
    @patch('chat.ConversationalAI')
    def test_main_function_initialization_error(self, mock_ai_class):
        """Test main function with initialization error."""
        mock_ai_class.side_effect = Exception("Init error")
        
        result = main()
        
        assert result == 1

    @patch('sys.argv', ['script.py', '--region', 'us-west-2', '--max-tokens', '256'])
    @patch('chat.ConversationalAI')
    @patch('builtins.input')
    @patch('builtins.print')
    def test_main_function_with_args(self, mock_print, mock_input, mock_ai_class):
        """Test main function with command line arguments."""
        mock_input.return_value = "quit"
        mock_ai = Mock()
        mock_ai_class.return_value = mock_ai
        
        result = main()
        
        assert result == 0
        # Verify AI was initialized with custom config
        args, kwargs = mock_ai_class.call_args
        config = args[0]
        assert config.region_name == "us-west-2"
        assert config.max_tokens == 256


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--cov=chat", "--cov-report=term-missing"])
