import tarfile
import tempfile
import requests
from fastapi import FastAPI
import gradio as gr
import time
import os
from pathlib import Path

app = FastAPI()

def update_interface(advanced_options):
    if advanced_options:
        return {additional_settings_textbox: gr.Textbox(visible=True)}
    else:
        return {additional_settings_textbox: gr.Textbox(visible=False)}

def run(module, inputs, request: gr.Request):
    if not request.query_params.get("userApiToken", "").startswith("lp-"):
        raise Exception("Please log in / register")
    res = requests.post("http://api/api/v1/jobs/sync", headers={
        "Authorization": "Bearer "+request.query_params["userApiToken"]
    }, json={
        "module": module,
        "inputs": inputs,
    })
    j = None
    try:
        j = res.json()
    except Exception as e:
        raise Exception(f"Error parsing JSON response: {res.text}")
    if j is not None:
        return getTarball(j["result"]["deal_id"])


def sdxl(prompt, model, size, request: gr.Request) -> Path:
    size_map = {
        "512 x 512": "512",
        "768 x 768": "768",
        "1024 x 1024": "1024",
        "1536 x 1536": "1536"
    }
    model_map = {
        "SDXL Base 0.9": "sdxl-pipeline:v0.9-base-lilypad1",
        "SDXL Refiner 0.9": "sdxl-pipeline:v0.9-refiner-lilypad1",
        "SDXL Base 1.0": "sdxl-pipeline:v1.0-base-lilypad1",
        "SDXL Refiner 1.0": "sdxl-pipeline:v1.0-refiner-lilypad1",
    }
    results_dir = run(model_map[model], {"Prompt": f"{prompt}", "Size": f"{size_map[size]}"}, request)
    return Path(results_dir+"/outputs/image-42.png")

def cowsay(message, request: gr.Request) -> str:
    results_dir = run("cowsay:v0.0.3", {"Message": message}, request)
    return open(results_dir+"/stdout").read()


def mistral7b(message, history, request: gr.Request):
    inst_str = ""
    for i, (q, a) in enumerate(history):
        inst_str += f"[INST]{q}[/INST]{a}"
        if i < len(history) - 1:
            inst_str += "\n"
    inst_str += f"[INST]{message}[/INST]"
    prompt = f"<s>{inst_str}"
    results_dir = run("mistral-7b-instruct:v0.1-lilypad7", {"PromptEnv": f"PROMPT={prompt}"}, request)
    return open(results_dir+"/stdout").read().split("[START]")[1].split("[/INST]")[-1]

# TODO: show the API call made to LilySaaS API in the UI, so users can see
# easily how to recreate it

# WHAT TO DO NEXT: make the functions above call the authenticated (for now)
# lilysaas API and actually work end-to-end

# If user is not logged in, make the frontend display a "log in to run this
# model" button with direct links to google and github OAuth

APPS = {
    "cowsay":
        gr.Interface(
            fn=cowsay,
            inputs=gr.Textbox(lines=2, placeholder="What would you like the cow to say?"),
            outputs="text",
            allow_flagging="never",
            css="footer {visibility: hidden}"
        ),
    "mistral7b":
        gr.ChatInterface(mistral7b,
            css="footer {visibility: hidden}"
        ),
}

# should never access this route directly
@app.get("/")
def read_main():
    return {"message": "here be dragons"}

# download results from the solver to a tmp directory, returning the path
def getTarball(id: str) -> str:
    solverUrl = os.getenv("SOLVER_URL") + f"/api/v1/deals/{id}/files"

    r = requests.get(solverUrl, allow_redirects=True)

    with tempfile.TemporaryDirectory(delete=False) as temp_dir:
        temp_file_path = os.path.join(temp_dir, "temp.tar")
        with open(temp_file_path, "wb") as temp_file:
            temp_file.write(r.content)
            print(f" ---> HERE IS A FILE: {temp_file.name} from {solverUrl}")
        with tarfile.open(temp_file_path, "r") as tar:
            tar.extractall(temp_dir)
        return temp_dir

with gr.Blocks() as sdxl_app:
    gr.Interface(
        fn=sdxl,
        inputs=[
            gr.Textbox(lines=2, placeholder="Enter prompt for Stable Diffusion XL"),
            gr.Dropdown(["SDXL Base 0.9", "SDXL Refiner 0.9", "SDXL Base 1.0", "SDXL Refiner 1.0"], value="SDXL Base 1.0", label="Model", info="Select the specific model you want to use"),
            gr.Dropdown(["512 x 512", "768 x 768", "1024 x 1024", "1536 x 1536"], value="1024 x 1024", label="Size", info="Select the size of the image you want to generate"),
        ],
        outputs="image",
        allow_flagging="never",
        css="footer {visibility: hidden}"
    )
    advanced_checkbox = gr.Checkbox(label="Advanced options", value=False)
    # Advanced settings unlocked when you click 
    additional_settings_textbox = gr.Textbox(label="Additional Settings", value="Debug A", visible=False)

    advanced_checkbox.change(fn=update_interface, inputs=advanced_checkbox, outputs=additional_settings_textbox)

# must match path nginx/noxy is proxying to (see docker-compose.yml)
CUSTOM_PATH = "/gradio"

# Mount the applications
for (app_name, gradio_app) in APPS.items():
    print("mounting app", app_name, "->", gradio_app)
    app.mount(CUSTOM_PATH + "/" + app_name, gr.routes.App.create_app(gradio_app))

app.mount(CUSTOM_PATH + "/" + "sdxl", gr.routes.App.create_app(sdxl_app))