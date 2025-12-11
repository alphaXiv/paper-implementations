def build_basic_prompt() -> str:
    return "Please provide a natural, plain text representation of the document, formatted in Markdown. Skip any headers and footers. For ALL mathematical expressions, use LaTeX notation with \( and \) for inline equations and \[ and \] for display equations. Convert any tables into Markdown format."


def build_openai_silver_data_prompt_no_document_anchoring(_base_text: str) -> str:
    return (
        "Below is the image of one page of a PDF document. "
        "Just return the plain text representation of this document as if you were reading it naturally.\n"
        "Turn equations into a LaTeX representation, and tables into markdown format. Remove the headers and footers, but keep references and footnotes.\n"
        "Read any natural handwriting.\n"
        "This is likely one page out of several in the document, so be sure to preserve any sentences that come from the previous page, or continue onto the next page, exactly as they are.\n"
        "If there is no text at all that you think you should read, you can output null.\n"
        "Do not hallucinate."
    )


def claude_response_format_schema() -> dict:
    return (
        {
            "name": "page_response",
            "description": "Extracts text from pdf's.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "primary_language": {
                        "type": ["string", "null"],
                        "description": "The primary language of the text using two-letter codes or null if there is no text at all that you think you should read.",
                    },
                    "is_rotation_valid": {
                        "type": "boolean",
                        "description": "Is this page oriented correctly for reading? Answer only considering the textual content, do not factor in the rotation of any charts, tables, drawings, or figures.",
                    },
                    "rotation_correction": {
                        "type": "integer",
                        "description": "Indicates the degree of clockwise rotation needed if the page is not oriented correctly.",
                        "enum": [0, 90, 180, 270],
                        "default": 0,
                    },
                    "is_table": {
                        "type": "boolean",
                        "description": "Indicates if the majority of the page content is in tabular format.",
                    },
                    "is_diagram": {
                        "type": "boolean",
                        "description": "Indicates if the majority of the page content is a visual diagram.",
                    },
                    "natural_text": {
                        "type": ["string", "null"],
                        "description": "The natural text content extracted from the page.",
                    },
                },
                "required": [
                    "primary_language",
                    "is_rotation_valid",
                    "rotation_correction",
                    "is_table",
                    "is_diagram",
                    "natural_text",
                ],
            },
        },
    )
