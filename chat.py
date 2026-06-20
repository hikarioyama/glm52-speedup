#!/usr/bin/env python3
"""Minimal terminal chat client for the local GLM-5.2 llama-server (OpenAI-compatible API).

Usage:
    python3 ~/projects/glm52-speedup/chat.py          # talk to 127.0.0.1:8080
    HOST=127.0.0.1 PORT=8080 python3 chat.py

Commands inside the chat:
    /reset   clear the conversation
    /quit    exit  (Ctrl-D or Ctrl-C also work)
Streams tokens as they arrive. Keeps multi-turn history.
"""
import json
import os
import sys
import urllib.request

HOST = os.environ.get("HOST", "127.0.0.1")
PORT = os.environ.get("PORT", "8080")
URL = f"http://{HOST}:{PORT}/v1/chat/completions"
TEMP = float(os.environ.get("TEMP", "0.7"))

messages = []
if os.environ.get("SYSTEM"):
    messages.append({"role": "system", "content": os.environ["SYSTEM"]})

# GLM-5.2 is a reasoning model. Thinking is OFF by default here (set THINK=1 to start with it on);
# toggle live with /think and /nothink.
enable_thinking = os.environ.get("THINK", "0") not in ("0", "", "false", "False")


def stream_reply():
    body = json.dumps({
        "messages": messages,
        "stream": True,
        "temperature": TEMP,
        "cache_prompt": True,
        "chat_template_kwargs": {"enable_thinking": enable_thinking},
    }).encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    acc = []            # final answer (content) only -> goes back into history
    in_think = False    # GLM-5.2 is a reasoning model: reasoning_content streams first (shown dimmed)
    with urllib.request.urlopen(req) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                obj = json.loads(data)
            except json.JSONDecodeError:
                continue
            delta = obj.get("choices", [{}])[0].get("delta", {})
            think = delta.get("reasoning_content")
            if think:
                if not in_think:
                    sys.stdout.write("\033[2m[thinking] ")  # dim
                    in_think = True
                sys.stdout.write(think)
                sys.stdout.flush()
            content = delta.get("content")
            if content:
                if in_think:
                    sys.stdout.write("\033[0m\n")  # end dim, blank line before the answer
                    in_think = False
                acc.append(content)
                sys.stdout.write(content)
                sys.stdout.flush()
    if in_think:
        sys.stdout.write("\033[0m")
    print()
    return "".join(acc)


def main():
    global enable_thinking
    print(f"GLM-5.2 chat @ {URL}  (temp={TEMP}, thinking={'on' if enable_thinking else 'off'})"
          "   /think /nothink /reset /quit")
    while True:
        try:
            user = input("\n\033[1;36myou>\033[0m ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user:
            continue
        if user in ("/quit", "/exit"):
            break
        if user in ("/nothink", "/think"):
            enable_thinking = (user == "/think")
            print(f"(thinking {'on' if enable_thinking else 'off'})")
            continue
        if user == "/reset":
            messages[:] = [m for m in messages if m["role"] == "system"]
            print("(conversation reset)")
            continue
        messages.append({"role": "user", "content": user})
        sys.stdout.write("\033[1;32mglm>\033[0m ")
        sys.stdout.flush()
        try:
            reply = stream_reply()
        except Exception as e:
            print(f"\n[error: {e}]  (is llama-server up on {HOST}:{PORT}?)")
            messages.pop()
            continue
        messages.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    main()
