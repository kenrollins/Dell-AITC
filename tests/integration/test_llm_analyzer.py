import pytest
import respx
import httpx

@pytest.mark.asyncio
async def test_error_handling(llm_analyzer, ai_categories):
    """Test error handling in LLM analyzer."""
    # Test invalid category
    with pytest.raises(ValueError):
        await llm_analyzer.verify_match(
            use_case_text="Sample text",
            category_name="NonexistentCategory",
            match_type="PRIMARY",
            confidence=0.5
        )

    # Test empty use case text
    result = await llm_analyzer.analyze_no_match("")
    assert result["reason_category"] == "UNCLEAR_DESC"
    assert result["confidence"] == 0.0
    assert result["reason"].startswith("Error")

@pytest.fixture
def mock_openai():
    """Mock OpenAI client for testing."""
    class MockOpenAI:
        def __init__(self):
            self.chat = self

        async def completions(self):
            return self

        async def create(self, model, messages, temperature, max_tokens):
            class MockResponse:
                def __init__(self, content):
                    self.choices = [type('Choice', (), {'message': type('Message', (), {'content': content})()})]
            return MockResponse("Test response")

    return MockOpenAI()

@pytest.mark.asyncio
async def test_error_handling_extended(llm_analyzer, mock_openai, ai_categories):
    """Test extended error handling scenarios."""
    # Replace the OpenAI client with our mock
    llm_analyzer.openai_client = mock_openai

    # Test timeout handling
    with respx.mock(assert_all_called=False) as respx_mock:
        respx_mock.post("https://api.openai.com/v1/chat/completions").mock(
            side_effect=httpx.TimeoutException("Request timed out")
        )
        result = await llm_analyzer.analyze_no_match("Sample text")
        assert result["reason"].startswith("Error")
        assert result["reason_category"] == "OTHER"
        assert result["confidence"] == 0.0

    # Test malformed API response
    with respx.mock(assert_all_called=False) as respx_mock:
        respx_mock.post("https://api.openai.com/v1/chat/completions").mock(
            return_value=httpx.Response(200, json={"choices": [{"message": {"content": None}}]})
        )
        result = await llm_analyzer.analyze_no_match("Sample text")
        assert result["reason"].startswith("Error")
        assert "Failed to parse OpenAI response" in result["reason"]
        assert result["confidence"] == 0.0 