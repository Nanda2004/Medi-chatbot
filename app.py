import io
import json
from typing import Dict, List, Literal, Optional

from fastapi import FastAPI, File, HTTPException, Request, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image
from pydantic import BaseModel, Field
from transformers import AutoModelForVision2Seq, AutoProcessor
import torch


MODEL_ID = "google/med-gemma-2b-vision-preview"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

processor: Optional[AutoProcessor] = None
model: Optional[AutoModelForVision2Seq] = None


class ReportResponse(BaseModel):
  findings: str
  impression: str
  differential_diagnoses: Optional[List[str]] = Field(default=None)
  urgency_level: Literal["Normal", "Mild Concern", "Concerning", "Urgent"]
  recommendations: str
  disclaimer: str = "This is AI-generated assistance, not a diagnosis or substitute for professional medical evaluation."


class ChatRequest(BaseModel):
  message: str = Field(..., min_length=1, max_length=800)


class ChatResponse(BaseModel):
  reply: str
  disclaimer: str = "This is AI-generated assistance, not a diagnosis or substitute for professional medical evaluation."


REPORT_PROMPT = """
You are Med-Gemma, a careful radiology assistant. Review the attached medical image and
respond ONLY with valid JSON using these keys:
findings, impression, differential_diagnoses (array), urgency_level (Normal/Mild Concern/Concerning/Urgent),
recommendations, disclaimer. Use hedged language (e.g., "may represent") and never prescribe medication.
If quality is poor state that and request better imaging.
"""

CHAT_PROMPT = """
You are a conversational triage assistant supporting clinicians. Respond empathetically in 2-3 sentences.
Remind users that guidance is informational and advise clinical follow-up when symptoms are significant.
Avoid naming medications.
User question:
"""


def build_dummy_report() -> ReportResponse:
  return ReportResponse(
    findings="Lung fields appear grossly clear though fine detail is limited by sample image quality.",
    impression="Mild nonspecific changes that may represent early inflammatory process; clinical correlation advised.",
    differential_diagnoses=[
      "Early infectious process",
      "Atypical inflammatory change",
      "Technique-related artifact"
    ],
    urgency_level="Mild Concern",
    recommendations="Recommend repeat imaging with optimized exposure parameters and correlation with vitals.",
    disclaimer="This is AI-generated assistance, not a diagnosis or substitute for professional medical evaluation."
  )


app = FastAPI(title="Medical Image Analyzer", version="1.0.0")
app.add_middleware(
  CORSMiddleware,
  allow_origins=["*"],
  allow_credentials=True,
  allow_methods=["*"],
  allow_headers=["*"]
)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
  return templates.TemplateResponse("index.html", {"request": request})


def ensure_model_loaded() -> None:
  global processor, model
  if processor is not None and model is not None:
    return
  try:
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForVision2Seq.from_pretrained(
      MODEL_ID,
      torch_dtype=DTYPE,
      use_safetensors=False
    )
    model.to(DEVICE)
    model.eval()
  except Exception as load_error:  # noqa: BLE001
    raise RuntimeError(f"Unable to load Med-Gemma weights: {load_error}") from load_error


def tensorize(inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
  routed = {}
  for key, value in inputs.items():
    if isinstance(value, torch.Tensor):
      if key == "pixel_values":
        routed[key] = value.to(DEVICE, dtype=DTYPE)
      else:
        routed[key] = value.to(DEVICE)
    else:
      routed[key] = value
  return routed


@app.post("/analyze", response_model=ReportResponse, summary="Analyze an uploaded medical image")
async def analyze_image(image: UploadFile = File(..., description="Medical image file")):
  if not image:
    raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Image file is required.")

  if image.content_type is None or not image.content_type.startswith("image/"):
    raise HTTPException(status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail="Only image uploads are supported.")

  image_bytes = await image.read()
  if not image_bytes:
    raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Supplied image appears empty.")

  try:
    structured_text = await generate_image_report(image_bytes)
  except RuntimeError as llm_error:
    raise HTTPException(status_code=500, detail=str(llm_error)) from llm_error

  return parse_report(structured_text)


@app.post("/chat", response_model=ChatResponse, summary="Chat-based follow up questions")
async def chat_endpoint(payload: ChatRequest):
  user_message = payload.message.strip()
  if not user_message:
    raise HTTPException(status_code=400, detail="Message cannot be empty.")

  try:
    reply_text = await generate_chat_reply(user_message)
  except RuntimeError as chat_error:
    raise HTTPException(status_code=500, detail=str(chat_error)) from chat_error

  return ChatResponse(reply=reply_text)


async def generate_image_report(image_bytes: bytes) -> str:
  ensure_model_loaded()
  assert processor and model  # quiet type-checkers

  try:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
  except Exception as image_error:  # noqa: BLE001
    raise RuntimeError("Unable to decode image. Please upload a valid JPEG/PNG.") from image_error

  inputs = processor(images=image, text=REPORT_PROMPT, return_tensors="pt")
  inputs = tensorize(inputs)

  try:
    with torch.no_grad():
      generated = model.generate(
        **inputs,
        max_new_tokens=512,
        temperature=0.2,
        do_sample=False
      )
    text = processor.batch_decode(generated, skip_special_tokens=True)[0]
  except Exception as inference_error:  # noqa: BLE001
    raise RuntimeError(f"Med-Gemma inference failed: {inference_error}") from inference_error

  if not text:
    raise RuntimeError("Model returned an empty response.")

  return text


async def generate_chat_reply(message: str) -> str:
  ensure_model_loaded()
  assert processor and model
  dummy_image = Image.new("RGB", (2, 2), color="white")
  prompt = f"{CHAT_PROMPT}\n{message}\nAssistant:"
  inputs = processor(images=dummy_image, text=prompt, return_tensors="pt")
  inputs = tensorize(inputs)

  try:
    with torch.no_grad():
      generated = model.generate(
        **inputs,
        max_new_tokens=200,
        temperature=0.4,
        do_sample=True,
      )
    text = processor.batch_decode(generated, skip_special_tokens=True)[0]
  except Exception as inference_error:  # noqa: BLE001
    raise RuntimeError(f"Chat assistant unavailable: {inference_error}") from inference_error

  cleaned = text.strip()
  if not cleaned:
    raise RuntimeError("Chat model returned an empty response.")

  return cleaned


def parse_report(structured_text: str) -> ReportResponse:
  structured_text = extract_json_block(structured_text)
  try:
    parsed = json.loads(structured_text)
  except json.JSONDecodeError as decode_error:
    raise HTTPException(status_code=500, detail="Model response was not valid JSON.") from decode_error

  try:
    return ReportResponse(
      findings=parsed["findings"],
      impression=parsed["impression"],
      differential_diagnoses=parsed.get("differential_diagnoses") or [],
      urgency_level=parsed["urgency_level"],
      recommendations=parsed["recommendations"],
      disclaimer=parsed.get(
        "disclaimer",
        "This is AI-generated assistance, not a diagnosis or substitute for professional medical evaluation."
      )
    )
  except KeyError as missing_field:
    raise HTTPException(status_code=500, detail=f"Missing field in model response: {missing_field}") from missing_field


def extract_json_block(text: str) -> str:
  start = text.find("{")
  end = text.rfind("}")
  if start != -1 and end != -1 and end > start:
    return text[start : end + 1]
  return text


if __name__ == "__main__":
  import uvicorn

  uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)


