import asyncio
import logging

from google import genai
from google.genai import types

from config import Config

log = logging.getLogger(__name__)

# Delays (seconds) between retries on the main model; the fallback model
# gets a single attempt after the main model is exhausted.
RETRY_DELAYS = (2, 4)
REQUEST_TIMEOUT_MS = 60_000

PROMPT = """\
Sen deneyimli bir ev aşçısısın. Ekteki fotoğraf(lar) bir buzdolabının içini gösteriyor.

Görünen malzemeleri belirle ve ağırlıklı olarak bu malzemeleri kullanan 3 farklı tarif öner.
Her mutfakta bulunan temel kiler malzemelerini de (sıvı yağ, tuz, karabiber, un, şeker,
yaygın baharatlar, makarna, pirinç) kullanabilirsin; bunları malzeme listesinde "(kiler)"
diye işaretle. Fotoğrafta görünmeyen başka malzeme ekleme.

Her tarif için şu şablonu kullan:

🍳 <numara>. <tarif adı>
⏱ <toplam süre> · <kaç kişilik>

Malzemeler:
• <malzeme ve yaklaşık miktar>

Yapılışı:
1. <adım>

Kurallar:
- Varsayılan olarak Türkçe yaz. Kullanıcının fotoğrafa eklediği not İngilizce ise tamamen İngilizce yaz.
- Kullanıcı fotoğrafla birlikte bir istek iletmişse (örn. "vejetaryen", "30 dakikada"), tarifleri buna göre uyarla.
- Fotoğraflarda tanınabilir yiyecek yoksa tarif uydurma; kibarca yiyecek göremediğini söyle.
- Yanıtı düz metin olarak yaz: yukarıdaki şablondaki emojiler ve madde işaretleri dışında
  Markdown (yıldız, alt çizgi, diyez) veya başka biçimlendirme işareti kullanma.
"""


class RecipeError(Exception):
    """Both the main and the fallback model failed."""


async def suggest_recipes(
    client: genai.Client,
    config: Config,
    images: list[tuple[bytes, str]],
    caption: str | None,
) -> str:
    contents: list = [PROMPT]
    contents.extend(
        types.Part.from_bytes(data=data, mime_type=mime_type)
        for data, mime_type in images
    )
    if caption:
        contents.append(f"Kullanıcının notu: {caption}")

    last_error: Exception | None = None
    for attempt in range(len(RETRY_DELAYS) + 1):
        try:
            return await _generate(client, config.main_model, contents)
        except Exception as exc:
            last_error = exc
            log.warning(
                "Main model %s failed (attempt %d/%d): %s",
                config.main_model, attempt + 1, len(RETRY_DELAYS) + 1, exc,
            )
            if attempt < len(RETRY_DELAYS):
                await asyncio.sleep(RETRY_DELAYS[attempt])

    log.warning("Main model exhausted, trying fallback %s", config.fallback_model)
    try:
        return await _generate(client, config.fallback_model, contents)
    except Exception as exc:
        log.error("Fallback model %s failed: %s", config.fallback_model, exc)
        raise RecipeError(
            f"{config.main_model}: {last_error}\n{config.fallback_model}: {exc}"
        ) from exc


async def _generate(client: genai.Client, model: str, contents: list) -> str:
    response = await client.aio.models.generate_content(
        model=model,
        contents=contents,
        config=types.GenerateContentConfig(
            temperature=0.7,
            http_options=types.HttpOptions(timeout=REQUEST_TIMEOUT_MS),
        ),
    )
    text = (response.text or "").strip()
    if not text:
        raise RuntimeError("model returned an empty response")
    return text
