import os
import subprocess
import httpx

from anthropic import Anthropic
from dotenv import load_dotenv
from openai import api_key

try:
    import readline
    # #143 UTF-8 backspace fix for macOS libedit
    readline.parse_and_bind('set bind-tty-special-chars off')
    readline.parse_and_bind('set input-meta on')
    readline.parse_and_bind('set output-meta on')
    readline.parse_and_bind('set convert-meta off')
    readline.parse_and_bind('set enable-meta-keybindings on')
except ImportError:
    pass

client = Anthropic(base_url=os.getenv("ANTHROPIC_BASE_URL"))

MODEL = os.environ["MODEL_ID"]

SYSTEM = f"You are a coding agent based at {os.getcwd}.Use bash to solve problems,don't ask,act."

TOOLS = [{
    "name":"bash",
    "description":"Run a shell command",
    "input_schema":{
        "type":"object",
        "properties":{"command":{"type":"string"}},
        "required":["command"]
    }
},{
    "name":"tavily-search",
    "description":"search relative questions and get answer",
    "input_schema":{
        "type":"object",
        "properties":{"query":{"type":"string","description":"search key words"}},
        "required":["query"]
    }
}
]

TOOL_HANDLERS = {
    "bash": lambda input: run_bash(input["command"]),
    "tavily-search": lambda input: tavily_search(input["query"])
}

def tavily_search(query:str)->str:
    try:
        response = httpx.post("https://api.tavily.com/search",
                            json={
                                "api_key":os.environ["TAVILY_API_KEY"],
                                "query":query,
                                "max_results":5
                                },
                            )
        results = response.json()["results"]
        return "\n".join(f"{r['title']}\n{r['url']}\n{r['content'][:200]}" for r in results)
    except :
        return "error in tavily_search"

def run_bash(command:str)->str:
    dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]
    if any(d in command for d in dangerous):
        return "dangerous command,rejected!"
    try:
        r = subprocess.run(command,shell=True,cwd=os.getcwd(),capture_output=True,text=True,timeout=120)
        out = (r.stdout + r.stderr).strip()
        return out[:50000] if out else "no output"
    except subprocess.TimeoutExpired:
        return "timeout"
    
def agent_loop(messages:list):
    while True:
        response = client.messages.create(model=MODEL,max_tokens=20000,system=SYSTEM,tools=TOOLS,messages=messages)
        messages.append({"role":"assistant","content":response.content})
        if response.stop_reason != "tool_use":
            return
        results = []
        for block in response.content:
            if block.type == "tool_use":
                handler = TOOL_HANDLERS.get(block.name)
                if handler is None:
                    output = f"unknown tool: {block.name}"
                else:
                    output = handler(block.input)
                results.append({
                    "type":"tool_result",
                    "tool_use_id":block.id,
                    "content":output
                })
