"""Web gateway channel — FastAPI + SSE for browser UI.

Provides a browser-accessible chat interface with:
- GET  /          — embedded single-page chat UI
- POST /api/chat  — accepts a message, streams SSE response tokens
- GET  /health    — liveness probe

SSE event types emitted on POST /api/chat:
  event: token   data: {"text": "..."}     — partial text token
  event: done    data: {"thread_id": "..."}  — stream finished
  event: error   data: {"message": "..."}  — error during generation
"""

from __future__ import annotations

import json
import logging
import uuid
from typing import Any, AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from langchain_core.messages import HumanMessage
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Embedded HTML UI
# ---------------------------------------------------------------------------

_HTML_UI = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>TitanClaw</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:system-ui,sans-serif;background:#0d1117;color:#e6edf3;
     display:flex;flex-direction:column;height:100vh}
header{padding:14px 20px;border-bottom:1px solid #21262d;font-weight:600;
       font-size:1.1rem;letter-spacing:.02em}
#log{flex:1;overflow-y:auto;padding:20px;display:flex;flex-direction:column;gap:12px}
.msg{max-width:72%;padding:10px 14px;border-radius:12px;line-height:1.5;
     white-space:pre-wrap;word-break:break-word}
.user{background:#1f6feb;align-self:flex-end;border-bottom-right-radius:4px}
.ai{background:#161b22;border:1px solid #21262d;align-self:flex-start;
    border-bottom-left-radius:4px}
.ai.streaming::after{content:'▋';animation:blink .7s step-end infinite}
@keyframes blink{50%{opacity:0}}
form{display:flex;gap:8px;padding:14px 20px;border-top:1px solid #21262d;background:#0d1117}
input{flex:1;background:#161b22;border:1px solid #30363d;border-radius:8px;
      padding:10px 14px;color:#e6edf3;font-size:.95rem;outline:none}
input:focus{border-color:#388bfd}
button{background:#1f6feb;border:none;border-radius:8px;padding:10px 18px;
       color:#fff;font-size:.95rem;cursor:pointer;white-space:nowrap}
button:hover{background:#388bfd}
button:disabled{opacity:.5;cursor:default}
</style>
</head>
<body>
<header>TitanClaw</header>
<div id="log"></div>
<form id="form">
  <input id="inp" type="text" placeholder="Type a message…" autocomplete="off">
  <button id="btn" type="submit">Send</button>
</form>
<script>
const log=document.getElementById('log');
const inp=document.getElementById('inp');
const btn=document.getElementById('btn');
let threadId=sessionStorage.getItem('titanclaw_thread')||null;

function addMsg(role,text=''){
  const d=document.createElement('div');
  d.className='msg '+role;
  d.textContent=text;
  log.appendChild(d);
  log.scrollTop=log.scrollHeight;
  return d;
}

document.getElementById('form').addEventListener('submit',async e=>{
  e.preventDefault();
  const content=inp.value.trim();
  if(!content)return;
  inp.value='';
  btn.disabled=true;
  addMsg('user',content);
  const aiEl=addMsg('ai');
  aiEl.classList.add('streaming');

  try{
    const res=await fetch('/api/chat',{
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({content,thread_id:threadId})
    });
    if(!res.ok){aiEl.textContent='Error: '+res.statusText;return;}

    const reader=res.body.getReader();
    const dec=new TextDecoder();
    let buf='',evtName='';

    while(true){
      const{done,value}=await reader.read();
      if(done)break;
      buf+=dec.decode(value,{stream:true});
      const lines=buf.split('\\n');
      buf=lines.pop();
      for(const line of lines){
        if(line.startsWith('event:')){evtName=line.slice(6).trim();}
        else if(line.startsWith('data:')){
          const raw=line.slice(5).trim();
          if(!raw)continue;
          try{
            const d=JSON.parse(raw);
            if(evtName==='token'&&d.text){aiEl.textContent+=d.text;}
            if(evtName==='done'&&d.thread_id){
              threadId=d.thread_id;
              sessionStorage.setItem('titanclaw_thread',threadId);
            }
            if(evtName==='error'){aiEl.textContent='Error: '+d.message;}
          }catch{}
          evtName='';
        }
      }
    }
  }catch(err){
    aiEl.textContent='Network error: '+err.message;
  }finally{
    aiEl.classList.remove('streaming');
    btn.disabled=false;
    inp.focus();
  }
});
inp.focus();
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Request/response models
# ---------------------------------------------------------------------------


class ChatRequest(BaseModel):
    content: str
    thread_id: str | None = None
    user_id: str = "web-user"


# ---------------------------------------------------------------------------
# Gateway
# ---------------------------------------------------------------------------


class WebGateway:
    """
    FastAPI-based web gateway for the browser UI.

    Unlike HttpChannel (which fits into the Channel abstraction), this gateway
    directly wraps the compiled LangGraph graph and uses ``astream_events``
    to deliver token-level SSE streaming to the browser.

    Usage::

        gw = WebGateway(graph, tool_registry, cors_origins=["*"])
        import uvicorn
        await uvicorn.Server(uvicorn.Config(gw.app, host="0.0.0.0", port=8000)).serve()
    """

    def __init__(
        self,
        graph: Any,
        tool_registry: Any,
        cors_origins: list[str] | None = None,
    ) -> None:
        self._graph = graph
        self._tool_registry = tool_registry
        self._cors_origins = cors_origins or ["*"]
        self.app = self._build_app()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _tool_defs(self) -> list[Any]:
        from titanclaw.state import ToolDefinition

        return [
            ToolDefinition(
                name=t.name,
                description=t.description,
                parameters=t.parameters_schema,
            )
            for t in self._tool_registry.list_tools()
        ]

    async def _sse_stream(self, content: str, thread_id: str) -> AsyncIterator[str]:
        """Yield SSE-formatted strings by streaming graph events."""
        state_input = {
            "messages": [HumanMessage(content=content)],
            "available_tools": self._tool_defs(),
        }
        graph_config = {"configurable": {"thread_id": thread_id}}

        try:
            async for event in self._graph.astream_events(
                state_input, config=graph_config, version="v2"
            ):
                kind = event.get("event")
                if kind == "on_chat_model_stream":
                    chunk = event.get("data", {}).get("chunk")
                    if chunk is None:
                        continue
                    # Handle both string and list content (Anthropic format)
                    text: str = ""
                    raw = chunk.content
                    if isinstance(raw, str):
                        text = raw
                    elif isinstance(raw, list):
                        text = "".join(
                            b.get("text", "") if isinstance(b, dict) else str(b)
                            for b in raw
                        )
                    if text:
                        yield f"event: token\ndata: {json.dumps({'text': text})}\n\n"

            yield f"event: done\ndata: {json.dumps({'thread_id': thread_id})}\n\n"

        except Exception as exc:  # noqa: BLE001
            logger.exception("Error streaming graph for thread %s", thread_id)
            yield f"event: error\ndata: {json.dumps({'message': str(exc)})}\n\n"

    # ------------------------------------------------------------------
    # App construction
    # ------------------------------------------------------------------

    def _build_app(self) -> FastAPI:
        app = FastAPI(title="TitanClaw Web Gateway", docs_url=None, redoc_url=None)

        app.add_middleware(
            CORSMiddleware,
            allow_origins=self._cors_origins,
            allow_methods=["GET", "POST"],
            allow_headers=["Content-Type"],
        )

        @app.get("/", response_class=HTMLResponse)
        async def ui() -> str:
            return _HTML_UI

        @app.post("/api/chat")
        async def chat(req: ChatRequest) -> StreamingResponse:
            thread_id = req.thread_id or str(uuid.uuid4())
            return StreamingResponse(
                self._sse_stream(req.content, thread_id),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "X-Accel-Buffering": "no",  # disable nginx buffering
                },
            )

        @app.get("/health")
        async def health() -> dict[str, str]:
            return {"status": "ok"}

        return app
