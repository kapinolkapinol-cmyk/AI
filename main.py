"""Prywatny, osobisty asystent AI. Sekrety konfiguruje się po stronie serwera."""
import hmac
import os
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from openai import APIConnectionError, APIStatusError, OpenAI, RateLimitError
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent
WEB_DIR = BASE_DIR / "static"
APP_ACCESS_CODE = os.environ.get("APP_ACCESS_CODE", "")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-6-astra")

app = FastAPI(title="Mój Asystent AI", docs_url=None, redoc_url=None)
app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")

class ChatRequest(BaseModel):
    messages: list[dict] = Field(min_length=1, max_length=24)

@app.get("/", include_in_schema=False)
def home():
    return FileResponse(WEB_DIR / "index.html")

@app.get("/health")
def health():
    return {"status": "ok"}

def verify_access(code: str):
    if len(APP_ACCESS_CODE) < 10:
        raise HTTPException(status_code=503, detail="Właściciel musi ustawić APP_ACCESS_CODE (minimum 10 znaków) w ustawieniach serwera.")
    if not hmac.compare_digest(code, APP_ACCESS_CODE):
        raise HTTPException(status_code=401, detail="Nieprawidłowy kod dostępu.")

@app.post("/api/chat")
def chat(request: ChatRequest, x_app_code: str = Header(default="")):
    verify_access(x_app_code)
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise HTTPException(status_code=503, detail="Właściciel musi dodać OPENAI_API_KEY w ustawieniach serwera.")

    history = []
    total_chars = 0
    for item in request.messages[-24:]:
        role = item.get("role")
        content = item.get("content")
        if role not in ("user", "assistant") or not isinstance(content, str):
            continue
        content = content.strip()
        total_chars += len(content)
        if content:
            history.append({"role": role, "content": content})
    if not history or history[-1]["role"] != "user":
        raise HTTPException(status_code=422, detail="Wiadomość użytkownika jest wymagana.")
    if total_chars > 30000:
        raise HTTPException(status_code=413, detail="Rozmowa jest za długa. Rozpocznij nowy czat.")

    instructions = (
        "Jesteś osobistym asystentem użytkownika. Odpowiadaj po polsku, jasno i konkretnie, "
        "chyba że użytkownik poprosi o inny język. Pomagaj w nauce, pisaniu, tłumaczeniu, "
        "planowaniu i codziennych problemach. Gdy pytanie wymaga aktualnych informacji, "
        "korzystaj z wyszukiwarki internetowej i podawaj źródła. Nie twierdź, że wykonałeś "
        "czynność poza czatem, jeśli jej faktycznie nie wykonałeś. Przed działaniem, które "
        "zmienia dane lub kontaktuje się z innymi, poproś o wyraźne potwierdzenie."
    )
    try:
        client = OpenAI(api_key=api_key)
        response = client.responses.create(
            model=OPENAI_MODEL,
            instructions=instructions,
            tools=[{"type": "web_search"}],
            input=history,
        )
        answer = response.output_text.strip()
        if not answer:
            raise HTTPException(status_code=502, detail="AI nie zwróciło tekstowej odpowiedzi. Spróbuj ponownie.")
        return {"answer": answer}
    except RateLimitError:
        raise HTTPException(status_code=429, detail="Limit lub środki API zostały wyczerpane. Sprawdź ustawienia konta API.")
    except APIConnectionError:
        raise HTTPException(status_code=502, detail="Nie udało się połączyć z usługą AI. Spróbuj ponownie.")
    except APIStatusError as exc:
        raise HTTPException(status_code=502, detail=f"Usługa AI zwróciła błąd (HTTP {exc.status_code}). Sprawdź model i dostęp API.")
