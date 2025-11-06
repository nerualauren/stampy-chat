"""Unit tests for tool execution."""
import pytest
from stampy_chat.llms import execute_calculator, execute_tool, TOOL_DEFINITIONS


class TestCalculator:
    """Test the calculator tool."""

    def test_simple_addition(self):
        assert execute_calculator("2 + 2") == "4"

    def test_multiplication(self):
        assert execute_calculator("123 * 456") == "56088"

    def test_complex_expression(self):
        assert execute_calculator("(10 + 5) * 3") == "45"

    def test_power(self):
        assert execute_calculator("2 ** 10") == "1024"

    def test_division(self):
        result = execute_calculator("10 / 4")
        assert result == "2.5"

    def test_modulo(self):
        assert execute_calculator("10 % 3") == "1"

    def test_invalid_characters(self):
        result = execute_calculator("import os")
        assert "Error" in result
        assert "Invalid characters" in result

    def test_syntax_error(self):
        result = execute_calculator("2 +* 3")
        assert "Error" in result


class TestToolExecution:
    """Test the tool execution framework."""

    def test_execute_calculator(self):
        result = execute_tool("calculator", {"expression": "10 + 20"})
        assert result == "30"

    def test_unknown_tool(self):
        result = execute_tool("nonexistent_tool", {})
        assert "Error" in result
        assert "Unknown tool" in result

    def test_tool_definitions_valid(self):
        """Ensure tool definitions have required fields."""
        for tool in TOOL_DEFINITIONS:
            assert "name" in tool
            assert "description" in tool
            assert "input_schema" in tool
            assert "type" in tool["input_schema"]
            assert "properties" in tool["input_schema"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
