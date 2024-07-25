from fastapi import FastAPI
from transformers import pipeline
from pydantic import BaseModel


app = FastAPI()

model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"

model = pipeline(
    "text-generation",
    model=model_id,
)

class TextGenerationRequest(BaseModel):
    prompt : str
    max_length : int = 1000

@app.post("/generate_text")
async def generate_text(request: TextGenerationRequest):
    result = model(request.prompt , max_length = request.max_length , num_return_sequences = 1)
    print(result)
    return {"generated_text": result[0]["generate_text"][-1]}

@app.get("/")
async def root():
    return {"message" : "hello!!"}

