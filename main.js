const http = require("http");
const https = require("https");
const { URL } = require("url");

const LOGGER_NAME = "kristal-caye-api";
const LOG_LEVEL = String(process.env.LOG_LEVEL || "INFO").toUpperCase();
const GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions";
const GROQ_MODEL = process.env.GROQ_MODEL || "openai/gpt-oss-20b";
const FALLBACK_REPLY = "Please Wait For a Moment We Will Return Later";
const WALK_IN_REPLY = `WALK-IN RATES
Day: P100 Adult / P80 Kids
Night: P150 Adult / P100 Kids

OPTIONAL PAID ITEMS
Small Kubo P300
Big Kubo P500
Long Table + 6 Chairs P250
Videoke P500
Cottage available`;
const MESSAGE_KEYS = ["message", "text", "user_message", "input"];
const SYSTEM_PROMPT = `You are the official AI assistant for KRISTAL CAYE H220 Resort.

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
Cottage available`;
const BODY_SIZE_LIMIT = 1024 * 1024;

const BOOKING_PATTERNS = [
  /\b(book|booking|reserve|reservation|reschedule)\b/i,
  /\b(magpareserve|magpa-reserve|pareserve|pa[- ]?reserve|reserba|mag[- ]?book|mag[- ]?reserve)\b/i,
  /\b(paano\s+mag[- ]?(book|reserve))\b/i,
  /\b(availability|available)\b.*\b(today|tomorrow|weekend|date|slot|room|swim|ba|po)\b/i,
  /\bmay\s+slot\b/i,
  /\bmay\s+bakante\b/i,
  /\bslot\b/i,
  /\bschedule\b/i,
];

const WALK_IN_PATTERNS = [
  /\bwalk[\s-]?in\b/i,
  /\bentrance\b/i,
  /\bhow much\b.*\b(walk[\s-]?in|entrance)\b/i,
  /\bmagkano\b.*\b(walk[\s-]?in|entrance)\b/i,
];

const LEVEL_ORDER = {
  DEBUG: 10,
  INFO: 20,
  WARNING: 30,
  ERROR: 40,
};

function log(level, message, error) {
  if ((LEVEL_ORDER[level] || LEVEL_ORDER.INFO) < (LEVEL_ORDER[LOG_LEVEL] || LEVEL_ORDER.INFO)) {
    return;
  }

  const timestamp = new Date().toISOString();
  const suffix = error ? ` ${error.stack || error.message || String(error)}` : "";
  console.log(`${timestamp} ${level} ${LOGGER_NAME} ${message}${suffix}`);
}

function serializeResponse(reply) {
  return { reply };
}

function sendJson(res, statusCode, payload) {
  const body = JSON.stringify(payload);
  res.writeHead(statusCode, {
    "Content-Type": "application/json; charset=utf-8",
    "Content-Length": Buffer.byteLength(body),
  });
  res.end(body);
}

function normalizeMessage(message) {
  return String(message).replace(/\s+/g, " ").trim().slice(0, 4000);
}

function coerceMessageValue(value) {
  if (typeof value === "string") {
    return normalizeMessage(value);
  }

  if (typeof value === "number" || typeof value === "boolean") {
    return normalizeMessage(String(value));
  }

  return "";
}

function extractMessageFromPayload(payload) {
  if (payload && typeof payload === "object" && !Array.isArray(payload)) {
    for (const key of MESSAGE_KEYS) {
      const message = coerceMessageValue(payload[key]);
      if (message) {
        return message;
      }
    }
    return "";
  }

  if (typeof payload === "string") {
    return normalizeMessage(payload);
  }

  return "";
}

function parseIncomingMessage(rawBody) {
  if (!rawBody || rawBody.length === 0) {
    return "";
  }

  const rawText = rawBody.toString("utf8").trim();
  if (!rawText) {
    return "";
  }

  try {
    const payload = JSON.parse(rawText);
    return extractMessageFromPayload(payload);
  } catch (error) {
    if (error instanceof SyntaxError) {
      // Accept plain text bodies, but treat broken JSON objects/arrays as invalid.
      if (rawText.startsWith("{") || rawText.startsWith("[")) {
        log("WARNING", "Invalid JSON body received");
        return "";
      }
      return normalizeMessage(rawText);
    }

    log("WARNING", "Request body could not be parsed", error);
    return "";
  }
}

function isBookingMessage(message) {
  return BOOKING_PATTERNS.some((pattern) => pattern.test(message));
}

function isWalkInMessage(message) {
  return WALK_IN_PATTERNS.some((pattern) => pattern.test(message));
}

function extractReplyText(data) {
  if (!data || typeof data !== "object") {
    throw new Error("Groq response is invalid");
  }

  const { choices } = data;
  if (!Array.isArray(choices) || choices.length === 0) {
    throw new Error("Groq response missing choices");
  }

  const firstChoice = choices[0];
  if (!firstChoice || typeof firstChoice !== "object") {
    throw new Error("Groq choice is invalid");
  }

  const message = firstChoice.message;
  if (!message || typeof message !== "object") {
    throw new Error("Groq message is invalid");
  }

  const { content } = message;
  if (typeof content === "string") {
    return content
      .split(/\r?\n/)
      .map((line) => line.replace(/\s+/g, " ").trim())
      .filter(Boolean)
      .join("\n")
      .trim();
  }

  if (Array.isArray(content)) {
    const parts = [];

    for (const item of content) {
      if (!item || typeof item !== "object") {
        continue;
      }

      if (typeof item.text === "string" && item.text.trim()) {
        parts.push(item.text.trim());
      }
    }

    return parts.join("\n").trim();
  }

  throw new Error("Groq content is invalid");
}

function normalizeReply(reply) {
  const cleaned = String(reply)
    .split(/\r?\n/)
    .map((line) => line.replace(/\s+/g, " ").trim())
    .filter(Boolean)
    .join("\n")
    .trim();
  const fallbackCandidate = cleaned.replace(/^"+|"+$/g, "").trim();

  if (fallbackCandidate === FALLBACK_REPLY) {
    return FALLBACK_REPLY;
  }

  return cleaned || FALLBACK_REPLY;
}

function postJson(urlString, headers, body, timeoutMs) {
  const url = new URL(urlString);

  return new Promise((resolve, reject) => {
    const req = https.request(
      {
        method: "POST",
        protocol: url.protocol,
        hostname: url.hostname,
        port: url.port || undefined,
        path: `${url.pathname}${url.search}`,
        headers: {
          "Content-Type": "application/json",
          "Content-Length": Buffer.byteLength(body),
          ...headers,
        },
      },
      (res) => {
        let responseBody = "";
        res.setEncoding("utf8");
        res.on("data", (chunk) => {
          responseBody += chunk;
        });
        res.on("end", () => {
          resolve({
            statusCode: res.statusCode || 500,
            body: responseBody,
          });
        });
      },
    );

    req.setTimeout(timeoutMs, () => {
      req.destroy(new Error("Groq API request timed out"));
    });
    req.on("error", reject);
    req.write(body);
    req.end();
  });
}

async function generateGroqReply(userMessage, groqApiKey) {
  if (!groqApiKey) {
    return FALLBACK_REPLY;
  }

  const payload = JSON.stringify({
    model: GROQ_MODEL,
    temperature: 0,
    messages: [
      { role: "system", content: SYSTEM_PROMPT },
      { role: "user", content: userMessage },
    ],
  });

  try {
    const response = await postJson(
      GROQ_API_URL,
      { Authorization: `Bearer ${groqApiKey}` },
      payload,
      20000,
    );

    if (response.statusCode < 200 || response.statusCode >= 300) {
      log("ERROR", `Groq API returned ${response.statusCode}: ${response.body}`);
      return FALLBACK_REPLY;
    }

    const data = JSON.parse(response.body);
    return normalizeReply(extractReplyText(data));
  } catch (error) {
    log("ERROR", "Groq API request failed", error);
    return FALLBACK_REPLY;
  }
}

function readRequestBody(req) {
  return new Promise((resolve) => {
    const chunks = [];
    let totalSize = 0;
    let tooLarge = false;

    req.on("data", (chunk) => {
      if (tooLarge) {
        return;
      }

      totalSize += chunk.length;
      if (totalSize > BODY_SIZE_LIMIT) {
        tooLarge = true;
        log("WARNING", "Request body exceeded size limit");
        return;
      }

      chunks.push(chunk);
    });

    req.on("end", () => {
      if (tooLarge) {
        resolve(Buffer.alloc(0));
        return;
      }

      resolve(Buffer.concat(chunks));
    });

    req.on("error", (error) => {
      log("ERROR", "Failed to read request body", error);
      resolve(Buffer.alloc(0));
    });
  });
}

async function handleChat(req, res, groqApiKey) {
  const rawBody = await readRequestBody(req);
  const message = parseIncomingMessage(rawBody);

  if (!message) {
    sendJson(res, 200, serializeResponse(FALLBACK_REPLY));
    return;
  }

  if (isBookingMessage(message)) {
    sendJson(res, 200, serializeResponse(FALLBACK_REPLY));
    return;
  }

  if (isWalkInMessage(message)) {
    sendJson(res, 200, serializeResponse(WALK_IN_REPLY));
    return;
  }

  const reply = await generateGroqReply(message, groqApiKey);
  sendJson(res, 200, serializeResponse(reply));
}

function createServer() {
  const groqApiKey = String(process.env.GROQ_API_KEY || "").trim();
  if (!groqApiKey) {
    log("WARNING", "GROQ_API_KEY is not set. The API will return fallback replies.");
  }

  return http.createServer(async (req, res) => {
    let pathname = "/";
    try {
      pathname = new URL(req.url || "/", "http://localhost").pathname;
    } catch (error) {
      log("WARNING", "Invalid request URL received", error);
    }

    if (req.method === "POST" && pathname === "/chat") {
      try {
        await handleChat(req, res, groqApiKey);
      } catch (error) {
        log("ERROR", "Unhandled application error", error);
        sendJson(res, 200, serializeResponse(FALLBACK_REPLY));
      }
      return;
    }

    sendJson(res, 404, { detail: "Not Found" });
  });
}

function startServer() {
  const port = Number.parseInt(process.env.PORT || "8000", 10);
  const server = createServer();

  server.listen(port, "0.0.0.0", () => {
    log("INFO", `KRISTAL CAYE H220 Resort API listening on port ${port}`);
  });
}

if (require.main === module) {
  startServer();
}

module.exports = {
  BOOKING_PATTERNS,
  FALLBACK_REPLY,
  MESSAGE_KEYS,
  WALK_IN_REPLY,
  createServer,
  extractMessageFromPayload,
  generateGroqReply,
  isBookingMessage,
  isWalkInMessage,
  normalizeMessage,
  normalizeReply,
  parseIncomingMessage,
};
