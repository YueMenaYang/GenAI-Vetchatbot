import os
import re
import openai
import getpass
from dotenv import load_dotenv
from typing import Annotated, Literal
from typing_extensions import TypedDict
from langchain_core.tools import tool
# from langgraph.cache.base import BaseCache 
from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.prebuilt import create_react_agent, InjectedState
from langgraph.types import Command, interrupt
from langchain_core.tools.base import InjectedToolCallId
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from langchain.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.messages import HumanMessage, AIMessage
import requests

from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

from prompt import system_prompt, relevant_prompt, check_zipcode_prompt, doctor_prompt, hospital_recommender_prompt

from data import (
    get_openai_key,
    load_faiss_index,
    create_conversational_chain,
    ROOT
)

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
google_api_key = os.getenv("GOOGLE_PLACES_API_KEY")

embeddings = OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key=api_key)
index_path = ROOT / "data" / "faiss_index"
vector_store = load_faiss_index(
    index_path,
    embeddings  
)

model = ChatOpenAI(model="gpt-4o")

class State(MessagesState):
    next: str
    zip_code: str
    pending_zipcode: bool

# members = ['doctor', 'hospital_recommender', 'check_relevant', 'check_zipcode']
members = ['doctor', 'hospital_recommender', 'check_zipcode']
options = members + ["FINISH"]

class Router(TypedDict):
    """Worker to route to next. If no workers needed, route to FINISH."""

    next: Literal[*options]

def supervisor_node(state: State) -> Command[Literal[*members, "__end__"]]:
    messages = [
        {"role": "system", "content": system_prompt},
    ] + state["messages"]

    # --- Extract last message content, depending on format ---
    last_message = state["messages"][-1].content

    zip_code_match = re.search(r"\b\d{5}\b", last_message)
    zip_code = zip_code_match.group(0) if zip_code_match else None

    ### Invoke the LLM
    response = model.with_structured_output(Router).invoke(messages)

    # goto = response["next"]
    # update = {"next": goto}
    goto = response.get("next")
    if goto is None:
        # st.error(f"Supervisor didn’t return a “next” field: {response}")
        return {}  
   
    update = {"next": goto}
    if zip_code:
        update["zip_code"] = zip_code

    if goto == "FINISH":
        return Command(goto=END, update={"next": "FINISH"})
    else:
        return Command(goto=goto, update=update)

class RelevantRouter(TypedDict):
    """Worker to route to supervisor or FINISH for relevant agent. If the message is irrelevant, route to FINISH."""

    next: Literal["supervisor", "FINISH"]

def relevant_node(state: State) -> Command[Literal["supervisor", "__end__"]]:
    messages = [
        {"role": "system", "content": relevant_prompt},
    ] + state["messages"]

    ### Invoke the LLM
    response = model.with_structured_output(RelevantRouter).invoke(messages)

    goto = response["next"]
    if goto == "supervisor":
        # Do nothing extra; just return to supervisor
        return Command(goto=goto)
    else:
        return Command(
            update={
                "messages": [
                    AIMessage(
                        content="This assistant is for veterinary and dog-related questions. Please ask something related to your dog’s health or care.",
                        name="check_irrelevant"
                    )
                ]
            },
            goto=END
        )

class CheckZipcodeRouter(TypedDict):
    """Worker to route to supervisor, doctor, or FINISH for check_zipcode agent."""

    next: Literal["supervisor", "doctor", "FINISH"]

def check_zipcode_node(state: State) -> Command[Literal["supervisor", "doctor", "__end__"]]:
    messages = [
        {"role": "system", "content": check_zipcode_prompt},
    ] + state["messages"]

    ### Invoke the LLM
    response = model.with_structured_output(CheckZipcodeRouter).invoke(messages)

    goto = response["next"]
    if goto == "FINISH":
        return Command(goto=END)
    elif goto == "supervisor":
        return Command(goto=goto, update={"pending_zipcode": False})
    else:
        return Command(goto=goto)


@tool
def diagnose_tool(query: str) -> str:
    """Retrieve veterinary information based on symptom."""
    #Create a retriever
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})

    qa_chain = RetrievalQA.from_chain_type(
      llm=ChatOpenAI(model_name="gpt-4o", openai_api_key=openai.api_key),
      retriever=retriever
    )

    answer = qa_chain.run(query)
    return answer


@tool
def hospital_locator_tool(zip_code: Annotated[str, "Zip code for finding the nearby hospitals"]) -> str:
    """Return all nearby vet hospitals with detailed information by ZIP code."""
    key = google_api_key
    url = f"https://maps.googleapis.com/maps/api/place/textsearch/json?query=vet+hospitals+near+{zip_code}&key={key}"
    r = requests.get(url).json()

    if r.get("status") != "OK":
        return "Could not fetch hospital data."

    results = r.get("results", [])
    if not results:
        return "No veterinary hospitals found nearby."

    hospitals_info = []
    for i, item in enumerate(results):
        name = item.get("name", "Unknown")
        address = item.get("formatted_address", "No address available")
        rating = item.get("rating", "N/A")
        user_ratings = item.get("user_ratings_total", "N/A")
        description = item.get("business_status", "")
        info = (
            f"{i + 1}. {name}\n"
            f"   Address: {address}\n"
            f"   Rating: {rating}★ ({user_ratings} reviews)\n"
            f"   Status: {description if description else 'N/A'}"
        )
        hospitals_info.append(info)

    return "Nearby veterinary hospitals:\n\n" + "\n\n".join(hospitals_info)

doctor_agent = create_react_agent(
    model=model,
    tools=[diagnose_tool],
    prompt=doctor_prompt
)

def doctor_node(state: State) -> Command[Literal["supervisor", "human"]]:
    pending_zipcode = state.get("pending_zipcode")
    # Call the researcher agent from here.
    result = doctor_agent.invoke(state)

    if pending_zipcode:
        return Command(
            update={
                "messages": [
                    AIMessage(content=result["messages"][-1].content,
                                name="doctor")
                ],
                "pending_zipcode": False
            },
            goto="human"
        )
    else:
        return Command(
            update={
                "messages": [
                    AIMessage(content=result["messages"][-1].content, name="doctor")
                ]
            },
            goto="supervisor"
        )

hospital_recommender_agent = create_react_agent(
    model=model,
    tools=[hospital_locator_tool],
    prompt=hospital_recommender_prompt
)

def hospital_recommender_node(state: State) -> Command[Literal["supervisor", "human"]]:
  zip_code = state.get("zip_code")

  if zip_code:
      # If ZIP code is available, proceed with recommending hospitals
      result = hospital_recommender_agent.invoke(state)
      return Command(
          update={
              "messages": [
                  AIMessage(content=result["messages"][-1].content, name="hospital_recommender")
              ]
          },
          goto="supervisor"
        )
  else:
      # If ZIP code is missing, ask user for it
      return Command(
          update={
              "messages": [
                  AIMessage(
                      content="To proceed with hospital recommendations, could you please provide your ZIP code?",
                      name="hospital_recommender"
                  )
              ],
              "pending_zipcode": True
          },
          goto="human"
      )


def human_node(
    state: State, config
) -> Command[Literal["supervisor"]]:
    """A node for collecting user input."""

    user_input = interrupt(value="Ready for user input.")

    return Command(
        update={
            "messages": [
                HumanMessage(content=user_input, name='user')
            ]
        },
        goto="supervisor",
    )


builder = StateGraph(State)
builder.add_edge(START, "supervisor")
builder.add_node("supervisor", supervisor_node)
# builder.add_node("check_relevant", relevant_node)
builder.add_node("check_zipcode", check_zipcode_node)
builder.add_node("doctor", doctor_node)
builder.add_node("hospital_recommender", hospital_recommender_node)
builder.add_node("human", human_node)

checkpointer = MemorySaver()
graph = builder.compile(checkpointer=checkpointer)