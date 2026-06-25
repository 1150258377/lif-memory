"""
LIF-Memory web tuner.

A zero-dependency local web UI for scanning an Obsidian vault, querying the
continuous problem field, preserving LIF voltages across turns, and keeping
ChatGPT-style session history in the vault.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import threading
import time
import urllib.parse
import urllib.request
from datetime import date, datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

DEFAULT_PARAMS = {
    "threshold": 5.0,
    "semantic_sigma": 0.55,
    "time_sigma": 30.0,
    "graph_alpha": 0.35,
    "graph_steps": 2,
    "fast_decay": 0.76,
    "slow_decay": 0.94,
    "fast_weight": 0.65,
    "slow_weight": 0.35,
}

PROVIDER_PRESETS = {
    "qwen": {"base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1", "model": "qwen-plus", "env": "DASHSCOPE_API_KEY"},
    "deepseek": {"base_url": "https://api.deepseek.com", "model": "deepseek-chat", "env": "DEEPSEEK_API_KEY"},
    "kimi": {"base_url": "https://api.moonshot.cn/v1", "model": "moonshot-v1-8k", "env": "MOONSHOT_API_KEY"},
    "zhipu": {"base_url": "https://open.bigmodel.cn/api/paas/v4", "model": "glm-4-flash", "env": "ZHIPUAI_API_KEY"},
}


class FieldState:
    def __init__(self, path: Path):
        self.path = path
        self.vault_hash = ""
        self.note_count = 0
        self.note_vectors: list[dict] = []
        self.voltage: dict[str, dict] = {}
        self.llm_history: list[dict] = []
        self.evolution_log: list[dict] = []
        self.scan_ts = ""
        self.load()

    def load(self) -> None:
        if not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            return
        self.vault_hash = data.get("vault_hash", "")
        self.note_count = int(data.get("note_count", 0))
        self.note_vectors = data.get("note_vectors", [])
        self.voltage = data.get("voltage", {})
        self.llm_history = data.get("llm_history", [])
        self.evolution_log = data.get("evolution_log", [])
        self.scan_ts = data.get("scan_ts", "")

    def save(self) -> None:
        self.path.write_text(json.dumps({
            "vault_hash": self.vault_hash,
            "note_count": self.note_count,
            "note_vectors": self.note_vectors,
            "voltage": self.voltage,
            "llm_history": self.llm_history[-40:],
            "evolution_log": self.evolution_log[-200:],
            "scan_ts": self.scan_ts,
        }, ensure_ascii=False, indent=2), encoding="utf-8")

    def needs_rescan(self, vault: Path) -> bool:
        files = [p for p in vault.rglob("*.md") if p.is_file()]
        digest = hashlib.md5(str(sorted(str(p) for p in files)).encode()).hexdigest()
        return digest != self.vault_hash or len(files) != self.note_count

    def set_scan_result(self, vault: Path, note_vectors: list[dict]) -> None:
        files = [p for p in vault.rglob("*.md") if p.is_file()]
        self.vault_hash = hashlib.md5(str(sorted(str(p) for p in files)).encode()).hexdigest()
        self.note_count = len(note_vectors)
        self.note_vectors = note_vectors
        self.scan_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.save()

    def get_voltage(self, topic: str) -> tuple[float, float]:
        item = self.voltage.get(topic, {})
        return float(item.get("v_fast", 0.0)), float(item.get("v_slow", 0.0))

    def set_voltage(self, topic: str, v_fast: float, v_slow: float) -> None:
        self.voltage[topic] = {
            "v_fast": round(v_fast, 4),
            "v_slow": round(v_slow, 4),
            "last_date": date.today().isoformat(),
        }

    def push_llm(self, role: str, content: str) -> None:
        self.llm_history.append({"role": role, "content": content})
        self.llm_history = self.llm_history[-40:]

    def append_log(self, query: str, insight: str, changes: list[str]) -> None:
        self.evolution_log.append({
            "ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "query": query,
            "insight": insight,
            "changes": changes,
        })


class SessionStore:
    def __init__(self, path: Path):
        self.path = path
        self.sessions: dict[str, dict] = {}
        self.active_sid = ""
        self.load()
        if not self.active_sid or self.active_sid not in self.sessions:
            self.new_session()

    def load(self) -> None:
        if not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            return
        self.sessions = data.get("sessions", {})
        self.active_sid = data.get("active_sid", "")

    def save(self) -> None:
        self.path.write_text(json.dumps({"sessions": self.sessions, "active_sid": self.active_sid}, ensure_ascii=False, indent=2), encoding="utf-8")

    def new_session(self, title: str = "") -> dict:
        import uuid
        sid = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:4]
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.sessions[sid] = {"id": sid, "title": title or f"对话 {datetime.now().strftime('%m/%d %H:%M')}", "ts": now, "updated_ts": now, "messages": []}
        self.active_sid = sid
        self.save()
        return self.sessions[sid]

    def switch(self, sid: str) -> bool:
        if sid not in self.sessions:
            return False
        self.active_sid = sid
        self.save()
        return True

    def delete(self, sid: str) -> None:
        self.sessions.pop(sid, None)
        if self.active_sid == sid:
            self.active_sid = sorted(self.sessions.keys())[-1] if self.sessions else ""
        if not self.active_sid:
            self.new_session()
        else:
            self.save()

    def add_message(self, role: str, content: str, msg_type: str = "text") -> None:
        session = self.sessions.get(self.active_sid)
        if not session:
            return
        session["messages"].append({"role": role, "content": content, "type": msg_type, "ts": datetime.now().strftime("%H:%M:%S")})
        session["updated_ts"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if role == "user" and session["title"].startswith("对话 ") and len(session["messages"]) == 1:
            session["title"] = content[:20] + ("…" if len(content) > 20 else "")
        self.save()

    def messages(self, sid: str = "") -> list[dict]:
        return list(self.sessions.get(sid or self.active_sid, {}).get("messages", []))

    def list_sessions(self) -> list[dict]:
        out = []
        for session in self.sessions.values():
            messages = session.get("messages", [])
            preview = str(messages[-1].get("content", "")).replace("\n", " ") if messages else ""
            out.append({
                "id": session["id"],
                "title": session["title"],
                "ts": session["ts"],
                "updated_ts": session.get("updated_ts", session["ts"]),
                "count": len(messages),
                "preview": preview[:90],
                "active": session["id"] == self.active_sid,
            })
        return sorted(out, key=lambda item: item["updated_ts"], reverse=True)[:30]


class AppState:
    def __init__(self, vault: Path, llm_cfg: dict):
        self.vault = vault
        self.params_path = vault / "lif_field_params.json"
        self.state_path = vault / "lif_field_state.json"
        self.sessions_path = vault / "lif_sessions.json"
        self.log_path = vault / "LIF-Memory-进化日志.md"
        self.llm_cfg = llm_cfg
        self.params = self.load_params()
        self.field = FieldState(self.state_path)
        self.sessions = SessionStore(self.sessions_path)
        self.status = "idle"
        self.last_query = ""

    def load_params(self) -> dict:
        params = dict(DEFAULT_PARAMS)
        if self.params_path.exists():
            try:
                params.update(json.loads(self.params_path.read_text(encoding="utf-8")))
            except Exception:
                pass
        return params

    def save_params(self) -> None:
        self.params_path.write_text(json.dumps(self.params, ensure_ascii=False, indent=2), encoding="utf-8")


STATE: AppState | None = None


def call_llm(messages: list[dict]) -> str:
    cfg = STATE.llm_cfg
    if not cfg.get("api_key"):
        return "[未配置 API Key，LLM 功能不可用]"
    payload = {"model": cfg["model"], "messages": messages, "temperature": 0.7, "max_tokens": 1500}
    req = urllib.request.Request(
        cfg["base_url"].rstrip("/") + "/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Authorization": f"Bearer {cfg['api_key']}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return data["choices"][0]["message"]["content"]
    except Exception as exc:
        return f"[LLM 调用失败: {exc}]"


def scan_vault(vault: Path) -> int:
    STATE.status = "scanning"
    try:
        sys.path.insert(0, str(Path(__file__).parent))
        import continuous_problem_field as cpf
        notes = cpf.read_notes(vault, date.today(), 365, 3000, all_notes=True)
        cpf.diffuse_graph(notes, int(STATE.params.get("graph_steps", 2)), float(STATE.params.get("graph_alpha", 0.35)))
        serialized = []
        for note in notes:
            serialized.append({
                "rel_path": note.rel_path,
                "title": note.title,
                "day": note.day.isoformat(),
                "snippet": note.snippet,
                "tags": note.tags,
                "links": note.links,
                "vector": note.vector,
                "graph_vector": note.graph_vector,
                "centrality": note.centrality,
            })
        STATE.field.set_scan_result(vault, serialized)
        return len(serialized)
    finally:
        STATE.status = "idle"


def restore_notes():
    sys.path.insert(0, str(Path(__file__).parent))
    import continuous_problem_field as cpf
    notes = []
    for item in STATE.field.note_vectors:
        try:
            day = date.fromisoformat(item.get("day", "2000-01-01"))
        except Exception:
            day = date.today()
        note = cpf.NoteObservation(
            path=Path(item.get("rel_path", "")),
            rel_path=item.get("rel_path", ""),
            title=item.get("title", ""),
            day=day,
            text=item.get("snippet", ""),
            snippet=item.get("snippet", ""),
            tags=item.get("tags", []),
            links=item.get("links", []),
            vector=item.get("vector", {}),
        )
        note.graph_vector = item.get("graph_vector", {})
        note.centrality = item.get("centrality", 0.0)
        notes.append(note)
    return notes


def append_evolution_log(query: str, reply: str, voltage_info: str) -> None:
    block = f"\n## {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n**查询**：{query}\n\n**场状态**：{voltage_info}\n\n**洞察**：\n\n{reply}\n\n---\n"
    if not STATE.log_path.exists():
        STATE.log_path.write_text("# LIF-Memory 进化日志\n", encoding="utf-8")
    with open(STATE.log_path, "a", encoding="utf-8") as f:
        f.write(block)


def query_field(query: str) -> dict:
    sys.path.insert(0, str(Path(__file__).parent))
    import continuous_problem_field as cpf
    if not STATE.field.note_vectors:
        scan_vault(STATE.vault)
    notes = restore_notes()
    params = STATE.params
    hits, field_energy, daily_current, daily_completion = cpf.reconstruct_field(
        notes=notes,
        query=query,
        today=date.today(),
        top_k=8,
        time_sigma=float(params.get("time_sigma", 30.0)),
        semantic_sigma=float(params.get("semantic_sigma", 0.55)),
        all_notes=True,
    )
    topic = cpf.infer_topic(query)
    v_fast, v_slow = STATE.field.get_voltage(topic)
    threshold = float(params.get("threshold", 5.0))
    total_input = sum(daily_current.values())
    total_completion = sum(daily_completion.values())
    v_fast = max(0.0, v_fast * float(params.get("fast_decay", 0.76)) + total_input * 4.0 - total_completion)
    v_slow = max(0.0, v_slow * float(params.get("slow_decay", 0.94)) + total_input * 4.0 * 0.42 - total_completion * 0.30)
    v = float(params.get("fast_weight", 0.65)) * v_fast + float(params.get("slow_weight", 0.35)) * v_slow
    spiked = v >= threshold
    STATE.field.set_voltage(topic, v_fast, v_slow)

    evidence = []
    for idx, hit in enumerate(hits[:5], 1):
        evidence.append(f"{idx}. [{hit.note.title}] 相似={hit.semantic:.3f}\n   {hit.note.snippet}")
    evidence_text = "\n".join(evidence) if evidence else "（未找到相关证据，将基于问题本身讨论）"
    voltage_info = f"V={v:.3f} fast={v_fast:.3f} slow={v_slow:.3f} θ={threshold}"

    system_prompt = {"role": "system", "content": "你是 LIF-Memory 持续进化洞察引擎。基于知识库证据和历史对话，给出更深一层的洞察，并主动追问。"}
    user_content = f"【查询】{query}\n【场状态】{voltage_info}\n【证据】\n{evidence_text}\n\n请给出洞察，并提出一个下一步问题。"
    reply = call_llm([system_prompt] + STATE.field.llm_history[-20:] + [{"role": "user", "content": user_content}])

    STATE.field.push_llm("user", user_content)
    STATE.field.push_llm("assistant", reply)
    STATE.field.append_log(query, reply, [voltage_info])
    STATE.field.save()
    STATE.sessions.add_message("user", query)
    STATE.sessions.add_message("assistant", reply)
    append_evolution_log(query, reply, voltage_info)
    return {"reply": reply, "v_fast": v_fast, "v_slow": v_slow, "v": v, "spiked": spiked, "field_energy": field_energy}


HTML_PAGE = r"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>LIF-Memory Web Tuner</title><style>
:root{--bg:#0f172a;--panel:#111827;--panel2:#1f2937;--text:#e5e7eb;--muted:#94a3b8;--accent:#38bdf8;--border:rgba(148,163,184,.2)}*{box-sizing:border-box}body{margin:0;height:100vh;display:flex;flex-direction:column;background:var(--bg);color:var(--text);font-family:Inter,"Microsoft YaHei",sans-serif}header{height:56px;display:flex;align-items:center;gap:12px;padding:0 22px;border-bottom:1px solid var(--border);background:#0b1220}h1{font-size:17px;margin:0;color:#bae6fd}.chip{font-size:12px;color:var(--muted)}#statusChip{margin-left:auto}.layout{display:grid;grid-template-columns:286px 1fr 300px;min-height:0;flex:1}.sessions{background:#0b1220;border-right:1px solid var(--border);display:flex;flex-direction:column}.sessions-head{display:flex;gap:8px;align-items:center;padding:14px;border-bottom:1px solid var(--border)}.sessions-head strong{flex:1}.newbtn{background:rgba(56,189,248,.12);border:1px solid rgba(56,189,248,.35);color:#e0f7ff;border-radius:8px;padding:8px 10px;cursor:pointer}.search{padding:10px 12px;border-bottom:1px solid var(--border)}.search input{width:100%;background:#111827;border:1px solid var(--border);border-radius:8px;color:var(--text);height:34px;padding:0 10px}.sessions-list{overflow:auto;padding:8px}.sess{padding:10px 34px 10px 11px;border-radius:8px;position:relative;cursor:pointer;margin-bottom:6px}.sess:hover,.sess.active{background:#172033}.sess.active{outline:1px solid rgba(56,189,248,.38)}.sess-title{font-size:13px;font-weight:600;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.sess-preview{font-size:11.5px;color:var(--muted);white-space:nowrap;overflow:hidden;text-overflow:ellipsis;margin-top:4px}.sess-meta{font-size:10.5px;color:#64748b;margin-top:5px}.sess-del{position:absolute;right:7px;top:9px;background:transparent;border:0;color:var(--muted);cursor:pointer}.empty{padding:24px 12px;text-align:center;color:var(--muted);font-size:12px;line-height:1.7}.chat{display:flex;flex-direction:column;min-width:0;border-right:1px solid var(--border)}.msgs{flex:1;overflow:auto;padding:22px;display:flex;flex-direction:column;gap:13px}.msg{display:flex;gap:10px}.msg.user{flex-direction:row-reverse}.av{width:30px;height:30px;border-radius:50%;display:flex;align-items:center;justify-content:center;background:#2563eb;flex:none}.msg.user .av{background:#059669}.bub{max-width:78%;background:var(--panel2);border:1px solid var(--border);border-radius:12px;padding:11px 14px;font-size:13.5px;line-height:1.7;white-space:pre-wrap;word-break:break-word}.input{border-top:1px solid var(--border);padding:14px 18px;background:var(--panel)}.row{display:flex;gap:8px}.row input{flex:1;background:#0f172a;border:1px solid var(--border);border-radius:8px;color:var(--text);padding:10px 12px}.btn{border:1px solid var(--border);border-radius:8px;background:var(--panel2);color:var(--text);padding:9px 13px;cursor:pointer}.btn.primary{background:linear-gradient(135deg,#38bdf8,#a78bfa);color:#07111f;border:0}.side{padding:16px;overflow:auto}.card{border:1px solid var(--border);background:var(--panel);border-radius:10px;padding:12px;margin-bottom:10px}.k{font-size:11px;color:var(--muted)}.v{font-size:20px;font-weight:700;margin-top:4px}.bar{height:4px;background:#263244;border-radius:4px;overflow:hidden;margin-top:8px}.fill{height:100%;background:linear-gradient(90deg,#34d399,#fbbf24)}
</style></head><body><header><h1>⚡ LIF-Memory</h1><span class="chip">持续问题场网页端</span><span id="statusChip" class="chip">加载中</span></header><div class="layout"><aside class="sessions"><div class="sessions-head"><strong>历史对话</strong><button class="newbtn" onclick="newChat()">＋ 新对话</button></div><div class="search"><input id="sessionSearch" placeholder="搜索历史对话" oninput="filterSessions()"></div><div id="sessionsList" class="sessions-list"></div></aside><main class="chat"><div id="msgs" class="msgs"><div class="msg"><div class="av">⚡</div><div class="bub">你好，我是 LIF-Memory 网页调参器。输入问题后，我会扫描/查询你的 Obsidian 知识库，并把对话保存在左侧历史栏。</div></div></div><div class="input"><div class="row"><input id="qInput" placeholder="输入问题，例如：最近哪些问题在持续积累张力？" onkeydown="if(event.key==='Enter')doQuery()"><button id="queryBtn" class="btn primary" onclick="doQuery()">查询场</button><button class="btn" onclick="doScan()">重扫库</button></div></div></main><aside class="side"><div class="card"><div class="k">笔记数</div><div id="noteCount" class="v">-</div></div><div class="card"><div class="k">综合电位 V</div><div id="voltage" class="v">0.000</div><div class="bar"><div id="vbar" class="fill" style="width:0%"></div></div></div><div class="card"><div class="k">上次扫描</div><div id="scanTs" style="font-size:12px;margin-top:6px;color:var(--muted)">-</div></div><div class="card"><button class="btn" onclick="openLog()">显示日志路径</button></div></aside></div><script>
let params={},allSessions=[],activeSid='';
function esc(s){return String(s??'').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')}
function msg(role,content){const w=document.getElementById('msgs');const d=document.createElement('div');d.className='msg '+role;d.innerHTML=`<div class="av">${role==='user'?'你':'⚡'}</div><div class="bub">${esc(content).replace(/\n/g,'<br>')}</div>`;w.appendChild(d);w.scrollTop=w.scrollHeight}
async function status(){const r=await fetch('/api/status');const d=await r.json();params=d.params||{};document.getElementById('statusChip').textContent=d.status||'idle';document.getElementById('noteCount').textContent=d.note_count||0;document.getElementById('scanTs').textContent=d.scan_ts||'-'}
async function loadSessions(restore=false){const r=await fetch('/api/sessions');const d=await r.json();allSessions=d.sessions||[];activeSid=d.active||'';renderSessions(allSessions);if(restore&&activeSid){const r2=await fetch('/api/sessions/messages',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({sid:activeSid})});const m=await r2.json();restoreMessages(m.messages||[])}}
function renderSessions(list){const el=document.getElementById('sessionsList');el.innerHTML='';if(!list.length){el.innerHTML='<div class="empty">暂无历史对话<br>提问后会自动保存到这里。</div>';return}for(const s of list){const d=document.createElement('div');d.className='sess'+(s.active?' active':'');d.innerHTML=`<div class="sess-title">${esc(s.title||'未命名')}</div><div class="sess-preview">${esc(s.preview||'还没有消息')}</div><div class="sess-meta">${esc((s.updated_ts||s.ts||'').slice(5,16))} · ${s.count||0} 条</div><button class="sess-del" onclick="delSession(event,'${s.id}')">×</button>`;d.onclick=e=>{if(e.target.className==='sess-del')return;switchSession(s.id)};el.appendChild(d)}}
function filterSessions(){const q=document.getElementById('sessionSearch').value.toLowerCase().trim();renderSessions(!q?allSessions:allSessions.filter(s=>((s.title||'')+' '+(s.preview||'')).toLowerCase().includes(q)))}
function restoreMessages(list){document.getElementById('msgs').innerHTML='';if(!list.length){msg('assistant','新对话已开始，请输入你的问题。');return}for(const m of list)msg(m.role,m.content)}
async function newChat(){await fetch('/api/sessions/new',{method:'POST'});restoreMessages([]);await loadSessions()}
async function switchSession(sid){const r=await fetch('/api/sessions/switch',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({sid})});const d=await r.json();if(d.ok){restoreMessages(d.messages||[]);await loadSessions()}}
async function delSession(e,sid){e.stopPropagation();if(!confirm('删除这个对话？'))return;await fetch('/api/sessions/delete',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({sid})});await loadSessions(true)}
async function doScan(){msg('assistant','正在扫描知识库...');const r=await fetch('/api/scan',{method:'POST'});const d=await r.json();msg('assistant',`扫描完成：${d.note_count||0} 篇笔记。`);await status()}
async function doQuery(){const input=document.getElementById('qInput');const q=input.value.trim();if(!q)return;input.value='';msg('user',q);document.getElementById('queryBtn').disabled=true;msg('assistant','思考中...');const r=await fetch('/api/query',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({query:q})});const d=await r.json();document.getElementById('queryBtn').disabled=false;document.querySelector('#msgs .msg:last-child').remove();msg('assistant',d.reply||'[无回复]');const v=d.v||0,th=params.threshold||5;document.getElementById('voltage').textContent=Number(v).toFixed(3);document.getElementById('vbar').style.width=Math.min(100,v/th*100)+'%';await status();await loadSessions()}
async function openLog(){const r=await fetch('/api/log_path');const d=await r.json();msg('assistant','进化日志路径：\n'+d.path)}
status();loadSessions(true);setInterval(status,8000);setInterval(()=>loadSessions(),15000);
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def send_json(self, data: dict, status: int = 200) -> None:
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def read_body(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        return json.loads(self.rfile.read(length).decode("utf-8")) if length else {}

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        path = urllib.parse.urlparse(self.path).path
        if path in ("/", "/index.html"):
            body = HTML_PAGE.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif path == "/api/status":
            self.send_json({"status": STATE.status, "note_count": STATE.field.note_count, "scan_ts": STATE.field.scan_ts, "params": STATE.params, "voltage": STATE.field.voltage})
        elif path == "/api/log_path":
            self.send_json({"path": str(STATE.log_path)})
        elif path == "/api/sessions":
            self.send_json({"sessions": STATE.sessions.list_sessions(), "active": STATE.sessions.active_sid})
        else:
            self.send_response(404); self.end_headers()

    def do_POST(self):
        path = urllib.parse.urlparse(self.path).path
        body = self.read_body()
        if path == "/api/scan":
            thread = threading.Thread(target=lambda: scan_vault(STATE.vault), daemon=True)
            thread.start()
            for _ in range(120):
                if STATE.status == "idle" and STATE.field.note_count > 0:
                    break
                time.sleep(1)
            self.send_json({"note_count": STATE.field.note_count, "scan_ts": STATE.field.scan_ts})
        elif path == "/api/query":
            query = body.get("query", "").strip()
            if not query:
                self.send_json({"error": "empty"}, 400); return
            STATE.status = "thinking"
            try:
                result = query_field(query)
                self.send_json(result)
            except Exception as exc:
                self.send_json({"reply": f"[错误: {exc}]", "v": 0, "v_fast": 0, "v_slow": 0, "spiked": False})
            finally:
                STATE.status = "idle"
        elif path == "/api/sessions/new":
            self.send_json({"session": STATE.sessions.new_session(), "messages": []})
        elif path == "/api/sessions/switch":
            ok = STATE.sessions.switch(body.get("sid", ""))
            self.send_json({"ok": ok, "messages": STATE.sessions.messages(body.get("sid", "")) if ok else []})
        elif path == "/api/sessions/delete":
            STATE.sessions.delete(body.get("sid", ""))
            self.send_json({"ok": True, "sessions": STATE.sessions.list_sessions(), "active": STATE.sessions.active_sid})
        elif path == "/api/sessions/messages":
            self.send_json({"messages": STATE.sessions.messages(body.get("sid", ""))})
        else:
            self.send_response(404); self.end_headers()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LIF-Memory web tuner")
    parser.add_argument("--vault", type=Path, default=Path("."))
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--llm-provider", choices=list(PROVIDER_PRESETS.keys()), default="deepseek")
    parser.add_argument("--llm-api-key", default=None)
    parser.add_argument("--llm-model", default=None)
    return parser.parse_args()


def main() -> None:
    global STATE
    args = parse_args()
    vault = args.vault.resolve()
    preset = PROVIDER_PRESETS[args.llm_provider]
    api_key = args.llm_api_key or os.environ.get(preset["env"], "")
    local_cfg = Path(__file__).resolve().parent / "config" / "llm.local.json"
    if not api_key and local_cfg.exists():
        try:
            data = json.loads(local_cfg.read_text(encoding="utf-8"))
            api_key = data.get("api_keys", {}).get(args.llm_provider, "") or data.get("api_key", "")
        except Exception:
            pass
    STATE = AppState(vault, {"base_url": preset["base_url"], "model": args.llm_model or preset["model"], "api_key": api_key})
    if STATE.field.needs_rescan(vault):
        threading.Thread(target=lambda: scan_vault(vault), daemon=True).start()
    url = f"http://127.0.0.1:{args.port}"
    print(f"LIF-Memory web tuner: {url}")
    print(f"Vault: {vault}")
    print(f"LLM: {args.llm_provider} / {STATE.llm_cfg['model']} / {'configured' if api_key else 'missing API key'}")
    try:
        import webbrowser
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    except Exception:
        pass
    HTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
