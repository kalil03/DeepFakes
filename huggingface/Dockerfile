FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Instalar dependências de sistema básicas
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates git build-essential \
    && rm -rf /var/lib/apt/lists/*

# Instalar Node.js 20 (para build do frontend)
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/*

# Copiar dependências Python e instalar
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copiar frontend e buildar React
COPY frontend /app/frontend
WORKDIR /app/frontend
RUN npm install && npm run build

# Voltar para raiz da aplicação e copiar restante do projeto
WORKDIR /app
COPY . /app

# Porta padrão do Hugging Face Spaces
EXPOSE 7860

# Comando de inicialização (HF define PORT=7860)
CMD ["python", "app.py"]

