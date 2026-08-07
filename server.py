import os, re, json, asyncio, time, logging
from typing import Any, Dict, List, Optional

import httpx
from fastapi import FastAPI
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("wrapper")

VLLM = os.environ.get("VLLM_URL", "http://127.0.0.1:8001")
BACKBONE = os.environ.get("MODEL_NAME", "Qwen/Qwen3-4B-Instruct-2507")
SERVED = os.environ.get("SERVED_MODEL_NAME", "kwriting-scorer")
DIMS = ["content", "organization", "expression"]

# 캘리브레이션: score = a * raw + b   (calibrate.py 결과로 교체)
# 2026-08-07 갱신. 구 규칙 최소제곱값(a<1)은 반올림 세계에서 모든 예측을
# 같은 정수 칸에 밀어넣어 organization Spearman을 0.42 -> 0.22로 깎았다.
# 아래는 raw_v2.jsonl 100건에 대한 제약 격자 탐색 결과(잠정).
#   제약: 출력값 3종 이상, 예측 표준편차 >= 정답 표준편차의 60%
#   5-fold 교차검증 기준 RMSE 0.7883 / Spearman 0.2920, 과적합 폭 +0.040.
CALIB = json.loads(os.environ.get("CALIB", json.dumps({
    "content":      {"a": 0.95, "b": -0.35},
    "organization": {"a": 0.95, "b":  0.45},
    "expression":   {"a": 0.55, "b":  1.30},
})))

# train 2000건 실측 평균 (폴백용)
FALLBACK = {"content": 3.278, "organization": 3.337, "expression": 3.673}

RUBRIC = {
"content": """- 글의 주장과 핵심 내용이 문제에 적절하게 대응하는가
- 근거가 충분하고 구체적인가
- 주장과 근거 사이의 논리적 연결이 타당한가""",
"organization": """- 서론, 본론, 결론의 구조가 드러나는가
- 문단 간 연결이 자연스러운가
- 논리 전개 순서가 일관적인가""",
"expression": """- 문장이 자연스럽고 이해하기 쉬운가
- 어휘 사용이 적절한가
- 맞춤법, 띄어쓰기, 문법, 주술 호응에 문제가 없는가""",
}

SCALE = """5점: 매우 우수함. 결함이 거의 없고 구체적 강점이 뚜렷함.
4점: 우수함. 경미한 약점은 있으나 기준을 전반적으로 잘 충족함.
3점: 보통. 장점과 약점이 함께 있으며 기준을 부분적으로 충족함.
2점: 미흡함. 주요 결함이 있어 기준 충족이 제한적임.
1점: 매우 미흡함. 기준을 거의 충족하지 못하거나 심각한 결함이 있음."""

BRACES = str.maketrans("", "", "{}")

app = FastAPI()
_ready = False


def parse_input(text: str):
    """평가 서버가 보낸 user 메시지에서 주제와 본문을 추출."""
    p = e = ""
    mp = re.search(r"\[prompt_text\]\s*(.*?)(?=\[essay_text\]|$)", text, re.S)
    me = re.search(r"\[essay_text\]\s*(.*)$", text, re.S)
    if mp:
        p = mp.group(1).strip()
    if me:
        e = me.group(1).strip()
    if not e:
        e = text.strip()
    return p, e


def score_prompt(dim, prompt_text, essay, analysis: str = ""):
    block = ""
    if analysis and dim in ANALYZED_DIMS:
        block = f"""
[사전 구조 분석]
아래는 이 글을 기계적으로 분해한 결과이다. 판단의 근거로 활용하되,
분석에 없는 내용은 원문을 직접 확인하라.
{analysis}
"""
    return f"""너는 한국어 논증적 글을 채점하는 엄격하고 일관된 평가자이다.

[평가 영역]{dim}
{RUBRIC[dim]}

[점수 기준]
{SCALE}

[글쓰기 주제]
{prompt_text}

[논증적 글]
{essay}
{block}
위 글의{dim} 영역 점수를 1~5 중 하나로 판단하라.
설명 없이 숫자 하나만 출력하라."""


ANALYSIS_MAXTOK = int(os.environ.get("ANALYSIS_MAXTOK", "420"))
USE_ANALYSIS = os.environ.get("USE_ANALYSIS", "1") == "1"
# 분석을 주입할 영역. expression은 국소 판단이라 현행 유지가 안전.
ANALYZED_DIMS = set(
    filter(None, os.environ.get("ANALYZED_DIMS", "content,organization").split(",")))


def analysis_prompt(prompt_text, essay):
    return f"""너는 한국어 논증적 글의 구조를 기계적으로 분해하는 분석기이다.
점수를 매기지 말고, 평가하는 말도 쓰지 마라. 글에 있는 사실만 옮겨 적어라.

[글쓰기 주제]
{prompt_text}

[논증적 글]
{essay}

아래 항목을 순서대로, 형식 그대로 채워라. 각 항목은 한 줄로 쓴다.
없으면 "없음"이라고 쓴다. 추측하지 말고 글에 있는 표현을 그대로 인용한다.

[문단수] (숫자만)
[서론] 도입 역할을 하는 첫 문단의 첫 문장을 그대로 인용
[주장] 글 전체의 중심 주장이 가장 뚜렷하게 드러난 문장을 그대로 인용
[근거] 주장을 뒷받침하는 근거를 최대 4개까지 세미콜론(;)으로 구분해 요약
[근거유형] 각 근거를 사례/통계/인용/일반론/개인경험 중 하나로 분류해 세미콜론으로 나열
[문단시작표현] 두 번째 문단부터 각 문단의 첫 3어절을 세미콜론으로 나열
[결론] 마지막 문단의 첫 문장을 그대로 인용
[결론재진술] 마지막 문단이 앞의 주장을 다시 언급하는가 (예/아니오)
[주제이탈] 주제와 직접 관련 없는 문단이 있는가 (있으면 그 문단의 첫 3어절, 없으면 없음)"""


ANALYSIS_KEYS = ["문단수", "서론", "주장", "근거", "근거유형",
                 "문단시작표현", "결론", "결론재진술", "주제이탈"]


def parse_analysis(text: str) -> str:
    """모델 출력에서 항목만 추려 정규화. 형식을 벗어나면 빈 문자열."""
    if not text:
        return ""
    lines = []
    for k in ANALYSIS_KEYS:
        m = re.search(rf"\[\s*{k}\s*\]\s*(.*)", text)
        if m:
            v = m.group(1).strip().translate(BRACES)
            v = re.sub(r"\s+", " ", v)[:300]
            if v:
                lines.append(f"- {k}: {v}")
    # 절반 미만만 잡히면 신뢰하지 않는다
    if len(lines) < len(ANALYSIS_KEYS) // 2:
        return ""
    return "\n".join(lines)


async def analyze(client, prompt_text, essay) -> str:
    if not USE_ANALYSIS:
        return ""
    try:
        msgs = [{"role": "user",
                 "content": analysis_prompt(prompt_text, essay)}]
        d = await call_vllm(client, msgs, ANALYSIS_MAXTOK)
        return parse_analysis(d["choices"][0]["message"]["content"])
    except Exception as ex:
        log.warning("analysis failed:%s", ex)
        return ""


def rationale_prompt(prompt_text, essay, scores):
    s = "\n".join(f"-{d}:{scores[d]:.1f}점" for d in DIMS)
    return f"""너는 한국어 논증적 글을 채점하는 평가자이다.
아래 글에 대해 이미 다음 점수가 확정되었다.

{s}

각 점수를 부여한 근거를 작성하라.

[작성 규칙]
- 각 영역의 평가 기준에만 맞는 근거를 쓴다. 다른 영역 기준을 섞지 않는다.
- 글에 실제로 있는 특정 문장, 표현, 논지, 문단 전개, 오류 양상을 짚는다.
- 글에 없는 내용을 지어내지 않는다.
- "대체로 괜찮다" 같은 상투적 총평을 쓰지 않는다.
- 확정된 점수와 어긋나지 않게 쓴다.
- 각 영역 2~3문장, 한국어.
- 중괄호 문자를 절대 쓰지 않는다.

[평가 기준]
content:{RUBRIC['content']}
organization:{RUBRIC['organization']}
expression:{RUBRIC['expression']}

[글쓰기 주제]
{prompt_text}

[논증적 글]
{essay}

아래 형식으로만 출력하라.
###content
(근거)
###organization
(근거)
###expression
(근거)"""


async def call_vllm(client, messages, max_tokens, temperature=0.0,
                    logprobs=False, top_logprobs=0, n=1):
    body = {
        "model": BACKBONE, "messages": messages,
        "max_tokens": max_tokens, "temperature": temperature,
        "top_p": 1.0, "seed": 42, "n": n,
    }
    if logprobs:
        body["logprobs"] = True
        body["top_logprobs"] = top_logprobs
    r = await client.post(f"{VLLM}/v1/chat/completions", json=body, timeout=180.0)
    r.raise_for_status()
    return r.json()


def expectation_from_logprobs(data) -> Optional[float]:
    """생성 토큰 중 첫 숫자(1~5) 위치의 top_logprobs로 기댓값 계산."""
    try:
        content = data["choices"][0]["logprobs"]["content"]
    except Exception:
        return None
    for tok in content:
        cands = {}
        for c in tok.get("top_logprobs", []):
            t = c["token"].strip()
            if t in ("1", "2", "3", "4", "5"):
                import math
                cands[int(t)] = max(cands.get(int(t), 0.0), math.exp(c["logprob"]))
        if len(cands) >= 2:
            z = sum(cands.values())
            if z > 0:
                return sum(k * v for k, v in cands.items()) / z
        if tok["token"].strip() in ("1", "2", "3", "4", "5"):
            return float(tok["token"].strip())
    return None


async def sample_fallback(client, messages) -> Optional[float]:
    """logprobs 실패 시: 온도 샘플링 8회 평균."""
    try:
        d = await call_vllm(client, messages, 4, temperature=0.8, n=8)
        vals = []
        for ch in d["choices"]:
            m = re.search(r"[1-5]", ch["message"]["content"])
            if m:
                vals.append(float(m.group()))
        if vals:
            return sum(vals) / len(vals)
    except Exception as ex:
        log.warning("fallback failed:%s", ex)
    return None


async def score_one(client, dim, prompt_text, essay, analysis="") -> float:
    msgs = [{"role": "user",
             "content": score_prompt(dim, prompt_text, essay, analysis)}]
    try:
        d = await call_vllm(client, msgs, 4, logprobs=True, top_logprobs=20)
        v = expectation_from_logprobs(d)
        if v is not None:
            return v
    except Exception as ex:
        log.warning("logprobs path failed (%s):%s", dim, ex)
    v = await sample_fallback(client, msgs)
    return v if v is not None else FALLBACK[dim]


def parse_rationales(text: str) -> Dict[str, str]:
    out = {}
    for d in DIMS:
        m = re.search(rf"###\s*{d}\s*(.*?)(?=###|$)", text, re.S | re.I)
        if m:
            out[d] = m.group(1).strip()
    return out


DEFAULT_R = {
    "content": "제시된 글의 주장과 근거의 관계를 검토하여 판단하였다.",
    "organization": "글의 문단 구조와 논리 전개 순서를 검토하여 판단하였다.",
    "expression": "문장의 자연스러움과 어휘 및 어법 사용을 검토하여 판단하였다.",
}


def clean(t: str, dim: str) -> str:
    t = (t or "").translate(BRACES).replace("```", "").strip()
    t = re.sub(r"\s+", " ", t)
    if len(t) < 10:
        return DEFAULT_R[dim]
    return t[:900]


# 마지막 요청의 raw 값을 항상 보관한다. 평가 서버는 /debug/last 를 호출하지 않으므로
# 켜 두어도 부작용이 없고, 제출 이미지에서도 환경변수 없이 raw 대조가 가능하다.
_DEBUG = {"on": True, "last": None}

# 평가 서버가 실수 출력을 반올림하므로, 반올림 후 값이 실제 제출 점수다.
ROUND_OUT = os.environ.get("ROUND_OUT", "1") == "1"


def calibrate(dim: str, v: float) -> float:
    a, b = CALIB[dim]["a"], CALIB[dim]["b"]
    x = max(1.0, min(5.0, a * v + b))
    return float(round(x)) if ROUND_OUT else round(x, 3)


class ChatReq(BaseModel):
    model: Optional[str] = None
    messages: List[Dict[str, Any]] = []
    max_tokens: Optional[int] = 2048
    temperature: Optional[float] = 0.0
    top_p: Optional[float] = 1.0
    seed: Optional[int] = 42
    stop: Optional[Any] = None
    n: Optional[int] = 1


@app.get("/health")
async def health():
    global _ready
    if _ready:
        return JSONResponse({"status": "ok"})
    try:
        async with httpx.AsyncClient(trust_env=False) as c:
            r = await c.get(f"{VLLM}/v1/models", timeout=5.0)
        if r.status_code == 200:
            _ready = True
            return JSONResponse({"status": "ok"})
    except Exception:
        pass
    return Response(status_code=503)


@app.get("/v1/models")
async def models():
    return JSONResponse({
        "object": "list",
        "data": [{"id": SERVED, "object": "model", "owned_by": "submission"}],
    })


@app.post("/v1/chat/completions")
async def chat(req: ChatReq):
    user_text = ""
    for m in req.messages:
        if m.get("role") == "user":
            c = m.get("content")
            user_text = c if isinstance(c, str) else " ".join(
                p.get("text", "") for p in c if isinstance(p, dict))
    prompt_text, essay = parse_input(user_text)

    scores, rats = dict(FALLBACK), {}
    raw_map = {}
    analysis = ""
    try:
        async with httpx.AsyncClient(trust_env=False) as client:
            # 1단계: 구조 분해 (전 영역 공용, 1회)
            analysis = await analyze(client, prompt_text, essay)
            # 2단계: 영역별 점수 (logprobs 기댓값) — 병렬
            raw = await asyncio.gather(
                *[score_one(client, d, prompt_text, essay, analysis)
                  for d in DIMS])
            for d, v in zip(DIMS, raw):
                raw_map[d] = v
                scores[d] = calibrate(d, v)
            # 3단계: 확정 점수를 조건으로 근거 생성
            rmsg = [{"role": "user",
                     "content": rationale_prompt(prompt_text, essay, scores)}]
            rd = await call_vllm(client, rmsg, 700)
            rats = parse_rationales(rd["choices"][0]["message"]["content"])
    except Exception as ex:
        log.exception("inference failed:%s", ex)

    obj = {d: {"score": scores[d], "rationale": clean(rats.get(d), d)}
           for d in DIMS}
    if _DEBUG.get("on"):
        _DEBUG["last"] = {"raw": raw_map, "analysis": analysis}
    return JSONResponse({
        "id": f"chatcmpl-{int(time.time()*1000)}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": SERVED,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant",
                        "content": json.dumps(obj, ensure_ascii=False)},
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    })


@app.get("/debug/last")
async def debug_last():
    return JSONResponse(_DEBUG.get("last") or {})
