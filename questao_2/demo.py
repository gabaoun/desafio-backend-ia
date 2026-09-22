import os

from langchain_core.language_models.fake_chat_models import FakeListChatModel

from questao_2.chatbot import PythonTutorBot


def run_demo() -> None:
    has_api_key = bool(os.getenv("OPENAI_API_KEY"))

    if has_api_key:
        bot = PythonTutorBot(model_name="gpt-4o")
    else:
        print("[INFO] OPENAI_API_KEY nao encontrada. Rodando com FakeListChatModel.")
        respostas_simuladas = [
            (
                "Para criar uma lista em Python:\n\n"
                "```python\n"
                "# Lista vazia\n"
                "itens = []\n\n"
                "# Lista com elementos\n"
                "frutas = ['maca', 'banana', 'laranja']\n"
                "```"
            ),
            (
                "Para modificar a lista:\n\n"
                "```python\n"
                "frutas.append('uva')      # insere no final\n"
                "frutas.insert(0, 'kiwi')   # insere no indice 0\n"
                "frutas.remove('banana')    # remove por valor\n"
                "ultimo = frutas.pop()      # remove e retorna ultimo\n"
                "```"
            ),
            (
                "Diferencas entre List e Tuple:\n\n"
                "1. Mutabilidade: List eh mutavel; Tuple eh imutavel.\n"
                "2. Sintaxe: `[1, 2]` vs `(1, 2)`.\n"
                "3. Performance: Tuples consomem menos memoria e sao hashable (podem ser chaves de dict)."
            ),
        ]
        bot = PythonTutorBot(llm=FakeListChatModel(responses=respostas_simuladas))

    perguntas = [
        "Como criar uma lista em Python?",
        "Como adicionar e remover elementos dessa lista?",
        "Qual a diferenca entre lista e tupla?",
    ]

    session_id = "recruiter_demo"
    print("=" * 60)
    print("Execucao da Questao 2 - Chatbot LangChain")
    print("=" * 60)

    for i, pergunta in enumerate(perguntas, 1):
        print(f"\n[Pergunta {i}]: {pergunta}")
        resposta = bot.responder(pergunta, session_id=session_id)
        print("-" * 50)
        print(f"[Resposta]:\n{resposta}")


if __name__ == "__main__":
    run_demo()
