import jsonschema

def is_tool_schema(obj: dict) -> bool:
    """
    Check if obj is a valid JSON schema describing a tool compatible with OpenAI's tool calling.
    Example valid schema:
    {
      "name": "get_current_weather",
      "description": "Get the current weather in a given location",
      "parameters": {
        "type": "object",
        "properties": {
          "location": {
            "type": "string",
            "description": "The city and state, e.g. San Francisco, CA"
          },
          "unit": {
            "type": "string",
            "enum": ["celsius", "fahrenheit"]
          }
        },
        "required": ["location"]
      }
    }
    """
    try:
        assert set(obj.keys()) == {'name', 'description', 'parameters'}
        assert isinstance(obj['name'], str)
        assert obj['name'].strip()
        assert isinstance(obj['description'], str)
        assert isinstance(obj['parameters'], dict)

        assert set(obj['parameters'].keys()) == {'type', 'properties', 'required'}
        assert obj['parameters']['type'] == 'object'
        assert isinstance(obj['parameters']['properties'], dict)
        assert isinstance(obj['parameters']['required'], list)
        assert set(obj['parameters']['required']).issubset(set(obj['parameters']['properties'].keys()))
    except AssertionError:
        return False
    try:
        jsonschema.validate(instance={}, schema=obj['parameters'])
    except jsonschema.exceptions.SchemaError:
        return False
    except jsonschema.exceptions.ValidationError:
        pass
    return True