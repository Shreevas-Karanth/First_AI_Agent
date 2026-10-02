import json
from datetime import date
from pathlib import Path

import ollama


MODEL = "qwen3:4b"
TASKS_FILE = Path(__file__).parent / "tasks.json"


# -------------------------------
# Tool 1: Read tasks
# -------------------------------
def get_tasks(status: str = "ALL") -> str:
    """
    Retrieve tasks from the local JSON file.

    Args:
        status: ALL, PENDING, or IN_PROGRESS.

    Returns:
        Tasks matching the requested status.
    """
    with TASKS_FILE.open("r", encoding="utf-8") as file:
        tasks = json.load(file)

    if status.upper() != "ALL":
        tasks = [
            task for task in tasks
            if task["status"] == status.upper()
        ]

    return json.dumps(tasks, indent=2)


# -------------------------------
# Tool 2: Get today's date
# -------------------------------
def get_today() -> str:
    """Return today's date in YYYY-MM-DD format."""
    return date.today().isoformat()


# -------------------------------
# Register available tools
# -------------------------------
AVAILABLE_TOOLS = {
    "get_tasks": get_tasks,
    "get_today": get_today,
}

TOOLS = [get_tasks, get_today]


# -------------------------------
# AI Agent
# -------------------------------
def run_agent(user_input: str) -> str:

    messages = [
        {
            "role": "system",
            "content": """
You are a helpful personal task assistant.

You can use tools to retrieve task information.
Use get_today when you need today's date.
Use get_tasks to retrieve task information.

Never invent task data.
Only report information returned by the tools.
Explain the results clearly and concisely.
"""
        },
        {
            "role": "user",
            "content": user_input
        }
    ]

    print(messages)

    # Limit tool iterations to prevent endless loops.
    for _ in range(5):

        response = ollama.chat(
            model=MODEL,
            messages=messages,
            tools=TOOLS
        )

        assistant_message = response.message

        print("\n--- Assistant Content ---")
        print(assistant_message.content)

        print("\n--- Tool Calls ---")
        print(assistant_message.tool_calls)

        print("\n--- Thinking ---")
        print(assistant_message.thinking)

        # Preserve the assistant's response and tool calls.
        messages.append(
            assistant_message.model_dump(exclude_none=True)
        )

        tool_calls = assistant_message.tool_calls or []

        print("Tool Calls")

        print(tool_calls)

        # No tool call means the agent has finished.
        if not tool_calls:
            return assistant_message.content or ""

        # Execute the requested Python tools.
        for tool_call in tool_calls:

            print("Tool Call---->")
            
            tool_name = tool_call.function.name
            arguments = tool_call.function.arguments


            print(tool_name)
            print(arguments)

            function = AVAILABLE_TOOLS.get(tool_name)

            if function is None:
                result = f"Error: Unknown tool {tool_name}"

            else:
                try:
                    result = function(**arguments)
                except Exception as error:
                    result = f"Tool error: {error}"

            messages.append({
                "role": "tool",
                "tool_name": tool_name,
                "content": str(result)
            })

    return "The agent reached its tool-call limit."


# -------------------------------
# Application entry point
# -------------------------------
if __name__ == "__main__":

    print("================================")
    print("     Python AI Task Agent")
    print("================================")
    print("Type 'exit' to quit.\n")

    while True:
        user_input = input("You: ").strip()

        if user_input.lower() in {"exit", "quit"}:
            print("Agent: Goodbye!")
            break

        if not user_input:
            continue

        try:
            answer = run_agent(user_input)
            print(f"\nAgent: {answer}\n")

        except Exception as error:
            print(f"\nError: {error}\n")
