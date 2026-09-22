"""
Questão 2: Chatbot com IA Generativa para Programação Python (LangChain + GPT-4).

Lógica de Implementação:
1. Recebimento de input textual do usuário.
2. Integração com LLM (OpenAI GPT-4) via LangChain Core (LCEL).
3. Gerenciamento de histórico e contexto multi-sessão (isolamento de memória).
"""

import os
from collections import defaultdict

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

# Instrução do sistema para garantir respostas técnicas e idiomáticas (PEP 8)
SYSTEM_PROMPT = (
    "Você é um engenheiro sênior e especialista na linguagem Python. "
    "Responda às dúvidas dos desenvolvedores com explicações técnicas diretas, "
    "código limpo aderente à PEP 8 e foco em performance e boas práticas."
)


class PythonTutorBot:
    def __init__(
        self,
        model_name: str = "gpt-4o",
        temperature: float = 0.2,
        llm: BaseChatModel | None = None,
    ) -> None:
        """
        Configura o pipeline conversacional do LangChain conectando o PromptTemplate,
        o LLM (GPT-4) e o parser de saída textual (LCEL: prompt | llm | output_parser).
        """
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.llm = llm or ChatOpenAI(
            model=model_name,
            temperature=temperature,
            api_key=self.api_key or "sk-placeholder",
        )

        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", SYSTEM_PROMPT),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{question}"),
            ]
        )

        # Cadeia declarativa do LangChain (LCEL)
        self.chain = self.prompt | self.llm | StrOutputParser()

        # Isolamento do histórico de conversas por identificador de sessão
        self._history: dict[str, list[BaseMessage]] = defaultdict(list)

    def get_history(self, session_id: str) -> list[BaseMessage]:
        """Retorna o histórico de mensagens da sessão informada."""
        return self._history[session_id]

    def clear_history(self, session_id: str) -> None:
        """Limpa o histórico de uma sessão específica."""
        if session_id in self._history:
            self._history[session_id].clear()

    def responder(self, question: str, session_id: str = "default") -> str:
        """
        Executa a cadeia conversacional injetando o histórico prévio da sessão,
        obtém a resposta do LLM e atualiza a memória com o novo turno de diálogo.
        """
        history = self.get_history(session_id)
        response_text = self.chain.invoke(
            {
                "question": question,
                "chat_history": history,
            }
        )
        # Atualização atômica do histórico
        history.append(HumanMessage(content=question))
        history.append(AIMessage(content=response_text))
        return response_text


def modo_interativo() -> None:
    """Loop CLI para interação via terminal."""
    bot = PythonTutorBot()
    session_id = "cli_session"
    print("Python AI Tutor (LangChain + GPT-4)")
    print("Digite 'sair' para encerrar.")
    print("-" * 50)

    while True:
        try:
            pergunta = input("\nVocê: ").strip()
            if not pergunta:
                continue
            if pergunta.lower() in {"sair", "exit", "quit"}:
                break
            resposta = bot.responder(pergunta, session_id=session_id)
            print(f"\nBot:\n{resposta}")
        except KeyboardInterrupt:
            break
        except Exception as err:
            print(f"\nErro: {err}")


if __name__ == "__main__":
    modo_interativo()
