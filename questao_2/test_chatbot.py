from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage, HumanMessage

from questao_2.chatbot import PythonTutorBot


def test_chatbot_initialization():
    bot = PythonTutorBot()
    assert bot.prompt is not None
    assert bot.chain is not None


def test_chatbot_conversation_flow_and_memory():
    fake_llm = FakeListChatModel(
        responses=[
            "Para criar uma lista use `minha_lista = []`.",
            "Para adicionar use `minha_lista.append('item')`.",
        ]
    )
    bot = PythonTutorBot(llm=fake_llm)

    res1 = bot.responder("Como criar lista em Python?", session_id="sess_1")
    assert "minha_lista = []" in res1

    res2 = bot.responder("Como adicionar itens nela?", session_id="sess_1")
    assert "minha_lista.append('item')" in res2

    history = bot.get_history("sess_1")
    assert len(history) == 4
    assert isinstance(history[0], HumanMessage)
    assert isinstance(history[1], AIMessage)
    assert isinstance(history[2], HumanMessage)
    assert isinstance(history[3], AIMessage)


def test_chatbot_session_isolation():
    fake_llm = FakeListChatModel(responses=["Resposta Sessao A", "Resposta Sessao B"])
    bot = PythonTutorBot(llm=fake_llm)

    bot.responder("Pergunta A", session_id="user_a")
    bot.responder("Pergunta B", session_id="user_b")

    assert len(bot.get_history("user_a")) == 2
    assert len(bot.get_history("user_b")) == 2
    assert bot.get_history("user_a")[0].content == "Pergunta A"
    assert bot.get_history("user_b")[0].content == "Pergunta B"
