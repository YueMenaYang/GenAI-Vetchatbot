import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import streamlit as st
import uuid
from langchain_core.messages import HumanMessage, AIMessage

from pmodel import graph, model, RelevantRouter, api_key, google_api_key
from prompt import relevant_prompt

# Initialize session state
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if "history" not in st.session_state:
    st.session_state.history = []

if "state" not in st.session_state:
    st.session_state.state = {
        "messages": [],
        "next": None,
        "zip_code": "",
        "pending_zipcode": False,
    }

st.title("🐾 VetChat: Dog Health Assistant")

# Display full history
for msg in st.session_state.state["messages"]:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.markdown(msg.content)
    elif isinstance(msg, AIMessage):
        name = getattr(msg, "name", "assistant")
        with st.chat_message("assistant"):
            st.markdown(f"**{name}:** {msg.content}")


# Handle input
if prompt := st.chat_input("How can I help your dog today?"):
    user_msg = HumanMessage(content=prompt)
    st.session_state.state["messages"].append(user_msg)

    # Show user message immediately
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # 2️⃣ build the relevance check
    # load_dotenv()
    # api_key = os.getenv("OPENAI_API_KEY")
    # google_api_key = os.getenv("GOOGLE_PLACES_API_KEY")
    # model = ChatOpenAI(model="gpt-4o", openai_api_key=api_key)
    # print("model set")
    # st.markdown("Checking relevance ...")
    with st.spinner("Checking relevance..."):
        relevance_messages = [
            {"role":"system", "content": relevant_prompt},
            *st.session_state.state["messages"]
        ]
        resp = model.with_structured_output(RelevantRouter).invoke(relevance_messages)

        # 3️⃣ if not about dogs, show the canned reply and stop
    if resp["next"] != "supervisor":
        ai_msg = AIMessage(
            content=(
            "This assistant is for veterinary and dog-related "
            "questions. Please ask something related to your dog’s "
            "health or care."
            ),
            name="check_irrelevant"
        )
        st.session_state.state["messages"].append(ai_msg)
        with st.chat_message("assistant"):
            st.markdown(ai_msg.content)
            
        with st.sidebar:
            if st.button("🔄 Reset Chat"):
                st.session_state.thread_id = str(uuid.uuid4())
                st.session_state.state = {
                    "messages": [],
                    "next": None,
                    "zip_code": "",
                    "pending_zipcode": False,
                }
                st.rerun()
        
        # skip graph entirely
        st.stop()
    
    with st.spinner("Thinking..."):
        
        for update in graph.stream(
            st.session_state.state,
            config={"thread_id": st.session_state.thread_id},
            stream_mode="updates"
        ):
            for node, result in update.items():
                if isinstance(result, dict) and "messages" in result:
                    new_msgs = result["messages"]
                    st.session_state.state["messages"].extend(new_msgs)
                    for msg in new_msgs:
                        if isinstance(msg, AIMessage):
                            name = getattr(msg, "name", "assistant")
                            with st.chat_message("assistant"):
                                st.markdown(f"**{name}:** {msg.content}")
                for k in ["next", "zip_code", "pending_zipcode"]:
                    if isinstance(result, dict) and k in result:
                        st.session_state.state[k] = result[k]

# Reset
with st.sidebar:
    if st.button("🔄 Reset Chat"):
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.state = {
            "messages": [],
            "next": None,
            "zip_code": "",
            "pending_zipcode": False,
        }
        st.rerun()