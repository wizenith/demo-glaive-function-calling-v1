from potassium import Potassium, Request, Response
from transformers import AutoModelForCausalLM , AutoTokenizer

app = Potassium("my_app")

# @app.init runs at startup, and loads models into the app's context
@app.init
def init():
    model = AutoModelForCausalLM.from_pretrained("sahil2801/test3", trust_remote_code=True).half().cuda()

    context = {
        "model": model
    }

    return context

# @app.handler runs for every call
@app.handler("/")
def handler(context: dict, request: Request) -> Response:
    prompt = request.json.get("prompt")
    
    tokenizer = AutoTokenizer.from_pretrained("sahil2801/test3", trust_remote_code=True)
    model = context.get("model")
    
    prompt = f"SYSTEM: You are an helpful assistant \nUSER: {prompt}\n ASSISTANT:"
    
    inputs = tokenizer(prompt,return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs,do_sample=True,temperature=0.5,max_new_tokens=100)
    
    result = tokenizer.decode(outputs[0],skip_special_tokens=True)
    print(result)

    return Response(
        json = {"outputs": result}, 
        status=200
    )

if __name__ == "__main__":
    app.serve()