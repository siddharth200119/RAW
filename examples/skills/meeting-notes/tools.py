"""Optional tools for the skill. RAW only imports this file with `load_skill_tools=True`."""
from RAW.models import Tool, ToolParam


def count_words(text: str) -> str:
    return str(len(text.split()))


TOOLS = [
    Tool(
        name="count_words",
        description="Count the words in a piece of text.",
        parameters=[ToolParam(name="text", type="string", description="The text to count.", required=True)],
        function=count_words,
    )
]
