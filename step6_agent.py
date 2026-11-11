# step6_agent.py
import sys
sys.stdout.reconfigure(encoding="utf-8")

from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent

from step5_tool import retriever_tool

# ---------- MIND : le LLM ----------
llm = ChatOllama(model="llama3.2", temperature=0)

# ---------- BODY : les outils ----------
tools = [retriever_tool]

SYSTEM_PROMPT = (
    "You are an assistant answering questions about the scientific paper "
    "'Attention Is All You Need'. "
    "Use retriever_tool to search the paper. Write search queries in ENGLISH. "
    "Rules: "
    "1) For questions about the text (numbers, explanations, results), use content_type='text'. "
    "2) For questions asking to SHOW a figure, use content_type='image'. "
    "3) For questions asking to show a figure AND explain it, make TWO searches: "
    "first content_type='image', then content_type='text'. "
    "4) If the user asks for PASSAGES similar to an image, use content_type='text'. "
    "5) You CANNOT see images. Never describe the content of an image. "
    "Only give its file path and page. "
    "6) Base your answer only on the retrieved results. Read them carefully: "
    "check that the language pair, number or section matches the question. "
    "7) Always mention the page number of your sources. "
    "8) If the question has nothing to do with the paper, answer directly "
    "without calling the tool. "
    "9) Answer in the language of the user's question."
)

# ---------- Graphe ReAct compilé ----------
agent = create_react_agent(llm, tools, prompt=SYSTEM_PROMPT)


def ask(question: str):
    print("\n" + "=" * 70)
    print("QUESTION :", question)
    print("=" * 70)
    for step in agent.stream(
        {"messages": [("user", question)]}, stream_mode="values"
    ):
        msg = step["messages"][-1]
        kind = msg.type
        if kind == "ai" and getattr(msg, "tool_calls", None):
            for tc in msg.tool_calls:
                print(f"\n[ACTION] appel de {tc['name']} avec {tc['args']}")
        elif kind == "tool":
           print(f"\n[OBSERVATION]\n{msg.content}")
        elif kind == "ai":
            print(f"\n[RÉPONSE FINALE]\n{msg.content}")


if __name__ == "__main__":
    ask("Explain how multi-head attention works.")