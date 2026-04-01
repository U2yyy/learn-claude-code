from http import client
import os
import subprocess

from anthropic import Anthropic
from dotenv import load_dotenv

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
MODEl = os.environ["MODEL_ID"]

SYSTEM = f"You are a coding agent based at {os.getcwd}.Use bash to solve problems,don't ask,act."

TOOLS = [{
    "name":"base",
    "description":"Run a shell command",
    "input_schema":{
        "type":"object",
        "properties":{"command":{"type":"string"}},
        "required":["command"]
    }
}]