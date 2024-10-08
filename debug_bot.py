import json
import os
import streamlit as st
from typing import List
from zhipuai import ZhipuAI
from prompt_templates import (
    SYSTEM_PROMPT_TEMPLATE,
    BOT_PROFILE,
    FRIEND_PROFILE,
    CONVERSATION_CASES,
)

MODEL_ID = "glm-4-plus"
TEMPERATURE = 0
TOP_P = 0.9
MAX_TOKENS = 512

# App title
st.set_page_config(page_title="调试分身Prompt")


def get_api_key(key_name: str) -> str:
    if key_name in os.environ:
        api_key = os.environ.get(key_name)
    else:
        api_key = st.secrets[key_name]
    return api_key


llm = ZhipuAI(api_key=get_api_key("ZHIPUAI_API_KEY"))

with st.sidebar:
    st.subheader("分身信息")
    overridden_bot_profile = st.text_area("重载分身信息")

    st.subheader("对方信息")
    overridden_friend_profile = st.text_area("重载对方信息")

    st.subheader("对话案例")
    overridden_cases = st.text_area("重载对话案例")

    st.subheader("重载System Prompt")
    overridden_system_prompt = st.text_area("重载System Prompt")

# Store LLM generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = []

# Display system prompt
with st.expander(
    "System prompt (clear the overriding system prompt to fall back to the default)"
):
    current_system_prompt = SYSTEM_PROMPT_TEMPLATE
    if (
        overridden_system_prompt is not None
        and len(overridden_system_prompt.strip()) > 0
    ):
        current_system_prompt = overridden_system_prompt
    st.text(current_system_prompt)

with st.expander("分身信息"):
    current_bot_profile = BOT_PROFILE
    if overridden_bot_profile is not None and len(overridden_bot_profile.strip()) > 0:
        current_bot_profile = overridden_bot_profile
    st.text(current_bot_profile)

with st.expander("对方信息"):
    current_friend_profile = FRIEND_PROFILE
    if (
        overridden_friend_profile is not None
        and len(overridden_friend_profile.strip()) > 0
    ):
        current_friend_profile = overridden_friend_profile
    st.text(current_friend_profile)

with st.expander("对话案例"):
    current_conversation_cases = CONVERSATION_CASES
    if overridden_cases is not None and len(overridden_cases.strip()) > 0:
        current_conversation_cases = overridden_cases
    st.text(current_conversation_cases)

# Display or clear chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])


def clear_chat_history():
    st.session_state.messages = []


st.sidebar.button("Clear Chat History", on_click=clear_chat_history)


def prepare_short_term_history(messages: List[dict]) -> str:
    if len(messages) <= 1:
        return ""

    turns = []

    for turn in messages[:-2]:
        if turn["role"] == "assistant":
            turns.append(f"你：{message['content']}")
        else:
            turns.append(f"对方：{message['content']}")

    return "\n".join(turns)


def prepare_user_message(
    bot_profile, friend_profile, conversation_cases, short_term_history, friend_message
):
    return (
        f"## 你的信息\n{bot_profile}\n"
        f"## 对方信息\n{friend_profile}\n"
        f"## 近期聊天历史\n{short_term_history}\n"
        f"## 真实对话案例\n{conversation_cases}\n"
        f"## 对方消息\n{friend_message}\n"
        f"Assistant:"
    )


def generate_response():
    system_prompt = SYSTEM_PROMPT_TEMPLATE

    if (
        overridden_system_prompt is not None
        and len(overridden_system_prompt.strip()) > 0
    ):
        system_prompt = overridden_system_prompt.strip()

    short_term_history = prepare_short_term_history(st.session_state.messages)

    bot_profile = BOT_PROFILE
    if overridden_bot_profile is not None and len(overridden_bot_profile.strip()) > 0:
        bot_profile = overridden_bot_profile.strip()

    friend_profile = FRIEND_PROFILE
    if (
        overridden_friend_profile is not None
        and len(overridden_friend_profile.strip()) > 0
    ):
        friend_profile = overridden_friend_profile.strip()

    conversation_cases = CONVERSATION_CASES
    if overridden_cases is not None and len(overridden_cases.strip()) > 0:
        conversation_cases = overridden_cases.strip()

    user_message = prepare_user_message(
        bot_profile=bot_profile,
        friend_profile=friend_profile,
        conversation_cases=conversation_cases,
        short_term_history=short_term_history,
        friend_message=st.session_state.messages[-1]["content"],
    )

    resp = llm.chat.completions.create(
        model=MODEL_ID,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        max_tokens=MAX_TOKENS,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        stream=False,
    )
    return resp.choices[0].message.content


def _try_to_parse_from_markdown(llm_output: str) -> str:
    lines = llm_output.split("\n")
    thought = None
    message = None

    for i, line in enumerate(lines):
        if line.find("Thought") != -1 or line.find("thought") != -1:
            thought = lines[i + 1].strip()
        elif line.find("Message") != -1 or line.find("message") != -1:
            message = lines[i + 1].strip()

    if thought is None or message is None:
        raise ValueError("Failed to parse from markdown")

    return message


def try_to_parse_model_output(llm_output: str) -> str:
    if llm_output.startswith("```json"):
        llm_output = llm_output[len("```json") :]
    elif llm_output.startswith("```"):
        llm_output = llm_output[len("```") :]

    if llm_output.endswith("```"):
        llm_output = llm_output[: len(llm_output) - len("```")]

    chat_response = json.loads(llm_output)

    if chat_response is not None:
        return chat_response["message"]

    if llm_output.startswith("#"):
        try:
            return _try_to_parse_from_markdown(llm_output)
        except Exception as exc:
            raise ValueError(
                f"Failed to parse from markdown in model output: {llm_output}"
            ) from exc


# User-provided prompt
if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

# Generate a new response if last message is not from assistant
if (
    len(st.session_state.messages) > 0
    and st.session_state.messages[-1]["role"] == "user"
):
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = generate_response()
            placeholder = st.empty()
            full_response = ""
            for item in response:
                full_response += item
            bot_message = try_to_parse_model_output(full_response)
            placeholder.markdown(bot_message)
    message = {"role": "assistant", "content": bot_message}
    st.session_state.messages.append(message)
