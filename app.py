from potassium import Potassium, Request, Response
from transformers import AutoModelForCausalLM , AutoTokenizer

app = Potassium("my_app")

# @app.init runs at startup, and loads models into the app's context
@app.init
def init():
    model = AutoModelForCausalLM.from_pretrained("sahil2801/test3", trust_remote_code=True).half().cuda()
    tokenizer = AutoTokenizer.from_pretrained("sahil2801/test3", trust_remote_code=True)

    context = {
        "model": model,
        "tokenizer": tokenizer
    }

    return context

# @app.handler runs for every call
@app.handler("/")
def handler(context: dict, request: Request) -> Response:
    tokenizer = context.get("tokenizer")
    model = context.get("model")

    user_prompt = request.json.get("prompt")
    system_prompt = request.json.get("system", "You are an helpful assistant")
    temperature = request.json.get("temperature", 0.5)
    max_new_tokens = request.json.get("max_new_tokens", 100)
    
    prompt = f"SYSTEM: {system_prompt} \nUSER: {user_prompt}\n ASSISTANT:"
    
    inputs = tokenizer(prompt,return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        do_sample=True,
        temperature=temperature,
        max_new_tokens=max_new_tokens
    )
    
    result = tokenizer.decode(outputs[0],skip_special_tokens=True)

    return Response(
        json = {"outputs": result}, 
        status=200
    )

if __name__ == "__main__":
    app.serve()
