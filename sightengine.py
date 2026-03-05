import os
from io import BytesIO
from typing import Optional, Dict, Any

import requests

"""
Integração simples com a API do Sightengine.

Este módulo expõe uma função principal:

    check_image(image_bytes: bytes) -> Optional[Dict[str, Any]]

Ela retorna um dicionário normalizado com os campos:
{
  "is_deepfake": bool,
  "deepfake_score": float,
  "is_ai_generated": bool,
  "ai_score": float,
  "generator_type": "diffusion"|"gan"|"other"|"none"
}

Se as credenciais não estiverem configuradas ou se a requisição falhar,
retorna None e o backend simplesmente omite o bloco "sightengine" na
resposta JSON.
"""


API_URL = "https://api.sightengine.com/1.0/check.json"


def _get_credentials() -> Optional[Dict[str, str]]:
    user = os.getenv("SIGHTENGINE_USER")
    secret = os.getenv("SIGHTENGINE_SECRET")
    if not user or not secret:
        return None
    return {"api_user": user, "api_secret": secret}


def is_configured() -> bool:
    return _get_credentials() is not None


def check_image(image_bytes: bytes) -> Optional[Dict[str, Any]]:
    """
    Envia a imagem para a API do Sightengine e retorna um resumo
    padronizado. Usa melhor esforço — se algo falhar, retorna None.
    """
    creds = _get_credentials()
    if creds is None:
        return None

    files = {"media": ("image.jpg", BytesIO(image_bytes), "image/jpeg")}
    # Modelos voltados para deepfake/IA generativa. A API real pode
    # usar nomes de modelos ligeiramente diferentes; aqui usamos a
    # melhor aproximação documentada.
    data = {
        **creds,
        "models": "deepfake,genai",
    }

    try:
        resp = requests.post(API_URL, data=data, files=files, timeout=8)
        resp.raise_for_status()
        payload = resp.json()
    except Exception:
        return None

    # Estrutura do payload depende do plano/modelos. Fazemos parsing
    # defensivo para extrair os scores principais.
    deepfake_score = None
    ai_score = None
    generator_type = "none"

    # Exemplos de campos possíveis (melhor esforço):
    # - payload.get("deepfake", {}).get("score")
    # - payload.get("ai", {}).get("score")
    deepfake_block = payload.get("deepfake") or payload.get("deepfakes")
    if isinstance(deepfake_block, dict):
        deepfake_score = float(deepfake_block.get("score", 0.0))

    ai_block = payload.get("ai") or payload.get("genai")
    if isinstance(ai_block, dict):
        ai_score = float(ai_block.get("score", 0.0))
        # alguns campos possíveis: "type": "diffusion"|"gan"|...
        gen_type = ai_block.get("type")
        if isinstance(gen_type, str):
            if gen_type.lower() in {"diffusion", "gan"}:
                generator_type = gen_type.lower()
            else:
                generator_type = "other"

    deepfake_score = float(deepfake_score or 0.0)
    ai_score = float(ai_score or 0.0)

    result: Dict[str, Any] = {
        "is_deepfake": deepfake_score >= 0.5,
        "deepfake_score": round(deepfake_score, 4),
        "is_ai_generated": ai_score >= 0.5,
        "ai_score": round(ai_score, 4),
        "generator_type": generator_type,
    }
    return result

