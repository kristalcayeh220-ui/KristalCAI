import logging
import os
import re
from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field


logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("kristal-caye-api")

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama3-70b-8192")
FALLBACK_REPLY = "Please Wait For a Moment We Will Return Later"
WALK_IN_REPLY = """WALK-IN RATES
Day: P100 Adult / P80 Kids
Night: P150 Adult / P100 Kids

OPTIONAL PAID ITEMS
Small Kubo P300
Big Kubo P500
Long Table + 6 Chairs P250
Videoke P500
Cottage available"""
SYSTEM_PROMPT = """You are the official AI assistant for KRISTAL CAYE H220 Resort.

Answer customer questions clearly, politely, and accurately using ONLY the official information below.

STRICT RULES
1. If the user message is about booking, reservation, scheduling, or availability, reply EXACTLY:
Please Wait For a Moment We Will Return Later

2. If the answer is not explicitly available in the official information below, reply EXACTLY:
Please Wait For a Moment We Will Return Later

3. Do not guess. Do not invent. Do not add details that are not listed.
4. Keep replies short and professional.
5. Respond in the same language as the user.
6. Always use the exact name KRISTAL CAYE H220 Resort.

OFFICIAL INFORMATION
- Email: kristalcayeh220@gmail.com
- Phone: 0956 066 1705
- Location: Tibangan Riles Zone 2, San Miguel, Bulacan
- Day Rate: P6,000 (9AM-5PM, 1 room)
- Night Rate: P7,000 (night swim, 1 room)
- 22 Hours: P12,000 (3 rooms)
- Walk-in Day: Adult P100, Kids P80
- Walk-in Night: Adult P150, Kids P100

OPTIONAL PAID ITEMS
- Small Kubo: P300
- Big Kubo: P500
- Long Table + 6 Chairs: P250
- Videoke: P500
- Cottage: available

AMENITIES AND NOTES
- Day Rate and Night Rate include 1 room
- 22 Hours includes 3 rooms
- 22 Hours can accommodate more than 10 people
- Rent stays include the main amenities
- Walk-in guests pay amenities/items separately
- Catering service: P1,000 extra

If the user asks about walk-in or entrance fees, reply in exactly this format:
WALK-IN RATES
Day: P100 Adult / P80 Kids
Night: P150 Adult / P100 Kids

OPTIONAL PAID ITEMS
Small Kubo P300
Big Kubo P500
Long Table + 6 Chairs P250
Videoke P500
Cottage available
"""

BOOKING_PATTERNS = [
    re.compile(r"\b(book|booking|reserve|reservation|reschedule)\b", re.IGNORECASE),
    re.compile(
        r"\b(magpareserve|magpa-reserve|pareserve|pa[- ]?reserve|reserba|mag[- ]?book|mag[- ]?reserve)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\b(paano\s+mag[- ]?(book|reserve))\b", re.IGNORECASE),
    re.compile(r"\b(availability|available)\b.*\b(today|tomorrow|weekend|date|slot|room|swim|ba|po)\b", re.IGNORECASE),
    re.compile(r"\bmay\s+slot\b", re.IGNORECASE),
    re.compile(r"\bmay\s+bakante\b", re.IGNORECASE),
    re.compile(r"\bslot\b", re.IGNORECASE),
    re.compile(r"\bschedule\b", re.IGNORECASE),
]

WALK_IN_PATTERNS = [
    re.compile(r"\bwalk[\s-]?in\b", re.IGNORECASE),
    re.compile(r"\bentrance\b", re.IGNORECASE),
    re.compile(r"\bhow much\b.*\b(walk[\s-]?in|entrance)\b", re.IGNORECASE),
    re.compile(r"\bmagkano\b.*\b(walk[\s-]?in|entrance)\b", re.IGNORECASE),
]


class ChatRequest(BaseModel):
    message: str = Field(..., max_length=4000)


class ChatResponse(BaseModel):
    reply: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    groq_api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not groq_api_key:
        logger.warning("GROQ_API_KEY is not set. The API will return fallback replies.")

    timeout = httpx.Timeout(20.0, connect=5.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        app.state.http_client = client
        app.state.groq_api_key = groq_api_key
        yield


app = FastAPI(
    title="KRISTAL CAYE H220 Resort API",
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
    lifespan=lifespan,
)


def serialize_response(reply: str) -> dict[str, str]:
    response = ChatResponse(reply=reply)
    if hasattr(response, "model_dump"):
        return response.model_dump()
    return response.dict()


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError,
) -> JSONResponse:
    logger.warning("Invalid request payload: %s", exc.errors())
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content=serialize_response(FALLBACK_REPLY),
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled application error: %s", exc)
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content=serialize_response(FALLBACK_REPLY),
    )


def is_booking_message(message: str) -> bool:
    return any(pattern.search(message) for pattern in BOOKING_PATTERNS)


def is_walk_in_message(message: str) -> bool:
    return any(pattern.search(message) for pattern in WALK_IN_PATTERNS)


def extract_reply_text(data: dict[str, Any]) -> str:
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("Groq response missing choices")

    first_choice = choices[0]
    if not isinstance(first_choice, dict):
        raise ValueError("Groq choice is invalid")

    message = first_choice.get("message")
    if not isinstance(message, dict):
        raise ValueError("Groq message is invalid")

    content = message.get("content")
    if isinstance(content, str):
        reply = "\n".join(
            re.sub(r"\s+", " ", line).strip()
            for line in content.splitlines()
            if line.strip()
        )
        return reply.strip()

    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if not isinstance(item, dict):
                continue
            text = item.get("text")
            if isinstance(text, str) and text.strip():
                parts.append(text.strip())
        return "\n".join(parts).strip()

    raise ValueError("Groq content is invalid")


def normalize_reply(reply: str) -> str:
    cleaned = "\n".join(
        re.sub(r"\s+", " ", line).strip()
        for line in reply.splitlines()
        if line.strip()
    ).strip()
    fallback_candidate = cleaned.strip('"').strip()
    if fallback_candidate == FALLBACK_REPLY:
        return FALLBACK_REPLY
    return cleaned or FALLBACK_REPLY


async def generate_groq_reply(
    user_message: str,
    client: httpx.AsyncClient,
    groq_api_key: str,
) -> str:
    if not groq_api_key:
        return FALLBACK_REPLY

    payload = {
        "model": GROQ_MODEL,
        "temperature": 0,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
    }
    headers = {"Authorization": f"Bearer {groq_api_key}"}

    try:
        response = await client.post(GROQ_API_URL, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
        return normalize_reply(extract_reply_text(data))
    except httpx.HTTPStatusError as exc:
        logger.error(
            "Groq API returned %s: %s",
            exc.response.status_code,
            exc.response.text,
        )
    except httpx.RequestError:
        logger.exception("Groq API request failed")
    except ValueError as exc:
        logger.error("Invalid Groq response: %s", exc)

    return FALLBACK_REPLY


@app.post("/chat", response_model=ChatResponse)
async def chat(payload: ChatRequest, request: Request) -> ChatResponse:
    message = " ".join(payload.message.split())
    if not message:
        return ChatResponse(reply=FALLBACK_REPLY)

    if is_booking_message(message):
        return ChatResponse(reply=FALLBACK_REPLY)

    if is_walk_in_message(message):
        return ChatResponse(reply=WALK_IN_REPLY)

    reply = await generate_groq_reply(
        user_message=message,
        client=request.app.state.http_client,
        groq_api_key=request.app.state.groq_api_key,
    )
    return ChatResponse(reply=reply)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
