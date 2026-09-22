.PHONY: help install lint format test run-api demo-chatbot demo-search clean

help:
	@echo "Comandos disponíveis:"
	@echo "  make install        Instala todas as dependências"
	@echo "  make lint           Executa verificação estática com Ruff"
	@echo "  make format         Formata o código com Ruff"
	@echo "  make test           Executa suite completa de testes com Pytest"
	@echo "  make run-api        Inicia API FastAPI na porta 8000"
	@echo "  make demo-chatbot   Executa demonstração do Chatbot (Questão 2)"
	@echo "  make demo-search    Executa demonstração de Busca Semântica (Questão 3)"

install:
	pip install --upgrade pip
	pip install -r requirements-dev.txt

lint:
	ruff check .

format:
	ruff format .
	ruff check --fix .

test:
	pytest -v --tb=short

run-api:
	uvicorn questao_1.main:app --reload --host 0.0.0.0 --port 8000

demo-chatbot:
	python -m questao_2.demo

demo-search:
	python -m questao_3.demo

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .coverage faiss_index biblioteca.db
