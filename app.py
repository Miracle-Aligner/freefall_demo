import streamlit as st
import asyncio
import json
from pydantic import BaseModel
from typing import List
from pathlib import Path

from llm import sync_call, stream, stream_by_mode, StreamMode, is_function_call


# Import the LLM methods from your code snippet
# e.g. from llm_methods import call, sync_call, stream, LLMMessage
# for illustration, we’ll re-use the LLMMessage definition here:


class LLMMessage(BaseModel):
    role: str
    content: str


# ----------------------------------------------------------------------------
# Utility functions for loading prompts and building chat prompt text
# ----------------------------------------------------------------------------


def load_system_prompt(file_path: str) -> str:
    """Utility to load a system prompt from a file."""
    return Path(file_path).read_text(encoding="utf-8").strip()


def build_chat_prompt_for_sync_call(chat_history: List[LLMMessage]) -> str:
    """
    Creates a single prompt string from the messages, which can be passed
    to sync_call. The provided sync_call function, in your snippet, only
    uses a single system-level 'prompt' string. So we combine everything here.
    """
    # Example concatenation approach (adjust formatting as needed)
    compiled = []
    for msg in chat_history:
        if msg.role == "system":
            compiled.append(f"[System]\n{msg.content}\n")
        elif msg.role == "user":
            compiled.append(f"[User]\n{msg.content}\n")
        elif msg.role == "assistant":
            compiled.append(f"[Assistant]\n{msg.content}\n")
    return "\n".join(compiled)


# ----------------------------------------------------------------------------
# Simple Freefall Chat
# ----------------------------------------------------------------------------
def init_simple_freefall_session():
    """
    Initializes the session state for the 'Simple Freefall' chat.
    Loads prompts/simple.txt as a system prompt and sets up chat history.
    """
    if "simple_freefall_history" not in st.session_state:
        system_prompt = load_system_prompt("prompts/simple.txt")
        st.session_state["simple_freefall_history"] = [
            LLMMessage(role="system", content=system_prompt)
        ]


def run_simple_freefall_chat():
    """
    Displays the UI for the Simple Freefall chat.
    Uses sync_call to communicate with the LLM.
    """
    init_simple_freefall_session()
    chat_history = st.session_state["simple_freefall_history"]

    # Display any existing conversation
    st.warning("BE CAREFUL: the story wouldn't be saved!", icon="⚠️")
    for msg in chat_history:
        if msg.role == "user":
            with st.chat_message("user"):
                st.write(f"{msg.content}")
        elif msg.role == "assistant":
            with st.chat_message("assistant"):
                st.write(f"{msg.content}")
        elif msg.role == "system":
            with st.chat_message("system"):
                st.write(f"{msg.content}")

    # User input text box
    chat_input = st.chat_input("Enter your message:", key="simple_input")
    if chat_input:
        if chat_input.strip():
            # Add user message
            chat_history.append(LLMMessage(role="user", content=chat_input))
            with st.chat_message("user"):
                st.write(f"{chat_input}")

            # Build the single prompt from the entire conversation
            prompt_text = build_chat_prompt_for_sync_call(chat_history)

            # Call the LLM (using your sync_call from the snippet)
            # The snippet’s sync_call returns either text or JSON, depending on response_type
            try:
                response = sync_call(
                    prompt_text,
                    "text",
                )

                if is_function_call(response):
                    chat_history.append(
                        LLMMessage(role="system", content="Function call")
                    )
                else:
                    chat_history.append(LLMMessage(role="assistant", content=response))
            except Exception as e:
                llm_response = f"Error: {e}"
                chat_history.append(LLMMessage(role="system", content=llm_response))

            # Force a re-run to display the new message
            st.rerun()


# ----------------------------------------------------------------------------
# Freefall with Plot Chat
# ----------------------------------------------------------------------------
def init_freefall_with_plot_session():
    """
    Initializes the session state for the 'Freefall with Plot' chat.
    We have two prompt files:
     1) prompts/with_plot.txt (for normal communication)
     2) prompts/generate_plot.txt (a technical prompt for generating the plot)
    """
    if "with_plot_history" not in st.session_state:
        # Main communication prompt
        main_prompt = load_system_prompt("prompts/with_plot.txt")
        st.session_state["with_plot_history"] = [
            LLMMessage(role="system", content=main_prompt)
        ]

    if "plot_outline" not in st.session_state:
        st.session_state["plot_outline"] = ""


def generate_plot(chat_history: List[LLMMessage]) -> str:
    """
    Tool function to call the LLM with generate_plot.txt + the chat history.
    The LLM should return some "plot description/outline" to us.
    We then store that in session_state["plot_outline"].
    """
    technical_prompt = load_system_prompt("prompts/generate_plot.txt")

    # Combine the technical prompt + partial conversation if desired
    # Or simply pass the entire conversation along with the generate_plot prompt
    combined = technical_prompt + "\n\n--- Chat History ---\n"
    for msg in chat_history:
        combined += f"{msg.role.upper()}: {msg.content}\n"

    # Make the LLM call
    try:
        plot_result = sync_call(combined, response_type="text", model="o3-mini")
    except Exception as e:
        plot_result = f"Error during generate_plot call: {e}"

    # Return the textual representation of the generated plot
    return plot_result


def run_freefall_with_plot_chat():
    """
    Manages the Freefall with Plot chat scenario.
    The user can talk with the system; if minimal story outline is set,
    we utilize the generate_plot() function as a 'tool call',
    and then continue the conversation with updated system prompt that
    includes a 'HIDDEN PLOT OUTLINE'.
    """
    init_freefall_with_plot_session()
    chat_history = st.session_state["with_plot_history"]

    # Display any existing conversation so far
    st.warning("BE CAREFUL: the story wouldn't be saved!", icon="⚠️")
    for msg in chat_history:
        if msg.role == "user":
            with st.chat_message("user"):
                st.write(f"{msg.content}")
        elif msg.role == "assistant":
            with st.chat_message("assistant"):
                st.write(f"{msg.content}")
        elif msg.role == "system":
            with st.chat_message("system"):
                st.write(f"{msg.content}")

    chat_input = st.chat_input("Enter your message:", key="with_plot_input")
    if chat_input:
        if chat_input.strip():
            # Add user message
            chat_history.append(LLMMessage(role="user", content=chat_input))
            with st.chat_message("user"):
                st.write(f"{chat_input}")

            prompt_text = build_chat_prompt_for_sync_call(chat_history)
            try:
                llm_settings = None
                if not st.session_state["plot_outline"]:
                    llm_settings = {
                        "tools": [
                            {
                                "type": "function",
                                "function": {
                                    "name": "call_generate_plot",
                                    "description": "Calls a function generate_plot() when INITIAL USER ENGAGEMENT is complete",
                                    "strict": True,
                                    "parameters": {
                                        "type": "object",
                                        "required": [],
                                        "properties": {},
                                        "additionalProperties": False,
                                    },
                                },
                            }
                        ]
                    }

                response = sync_call(
                    prompt_text, response_type="text", llm_settings=llm_settings
                )

                if is_function_call(response):
                    if not st.session_state["plot_outline"]:
                        plot_outline_data = generate_plot(chat_history)

                        # Update session state
                        st.session_state["plot_outline"] = plot_outline_data

                        # Now update the system prompt to contain the hidden plot data
                        # We'll find the system message (first in chat_history) and append
                        for msg in chat_history:
                            if msg.role == "system":
                                msg.content += f'\n\n10. HIDDEN PLOT OUTLINE:\n"""\n{plot_outline_data}\n"""'
                                break

                    response = sync_call(prompt_text, response_type="text")

                chat_history.append(LLMMessage(role="assistant", content=response))

            except Exception as e:
                response = f"Error: {e}"
                chat_history.append(LLMMessage(role="system", content=response))

            st.rerun()


def run_generate_branches():
    st.subheader("Generate Branches")

    # User enters a story outline
    story_outline = st.text_area("Enter your story outline:", value="", height=200)

    # Button to trigger the branches generation
    if st.button("Generate Branches"):
        if story_outline.strip():
            # 1) Load the system prompt
            try:
                # Replace with your own function if needed.
                with open("prompts/generate_branches.txt", encoding="utf-8") as f:
                    system_prompt = f.read().strip()
            except Exception as e:
                st.error(f"Error reading file: {e}")
                return

            # 2) Build the combined prompt text
            # Format: <system_prompt> + \n\nCHAT HISTORY: """{story_outline}"""
            combined_prompt = f'{system_prompt}\n\nCHAT HISTORY: """{story_outline}"""'

            # 3) Call the LLM with response_type="json_object"
            #    (assumes your sync_call method can return a JSON dictionary)
            try:
                result = sync_call(
                    combined_prompt, response_type="text", model="o3-mini-2025-01-31"
                )
            except Exception as e:
                st.error(f"Error during sync_call: {str(e)}")
                return

            # 4) Display the JSON result in a readable format
            st.markdown("## Branches JSON Result")
            st.json(result)
        else:
            st.warning("Please enter a story outline before generating branches.")


# ----------------------------------------------------------------------------
# Main Streamlit App
# ----------------------------------------------------------------------------
def main():
    st.set_page_config(layout="wide", page_title="Freefall Chat App")
    st.title("Freefall Chat")

    # Sidebar mode selection
    mode = st.sidebar.radio(
        "Select Chat Mode:",
        ("Simple Freefall", "Freefall with Plot", "Generate branches"),
    )

    if mode == "Simple Freefall":
        run_simple_freefall_chat()
    elif mode == "Freefall with Plot":
        run_freefall_with_plot_chat()
    elif mode == "Generate branches":
        run_generate_branches()


if __name__ == "__main__":
    main()
