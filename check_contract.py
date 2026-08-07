"""제출 계약 검증. Windows/Mac/Linux 어디서나 실행된다.

사용: python3 check_contract.py https://<POD_ID>-8000.proxy.runpod.net
      python3 check_contract.py                      (기본 http://127.0.0.1:8000)

Docker Image 제출 규정 12절 체크리스트 + FAQ의 출력 형식 규칙을 검사한다.
"""
import json, sys
import httpx

BASE = (sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000").rstrip("/")
DIMS = ["content", "organization", "expression"]
ok_n = fail_n = 0


def ok(m):
    global ok_n; ok_n += 1; print(f"  PASS  {m}")


def ng(m):
    global fail_n; fail_n += 1; print(f"  FAIL  {m}")


print(f"=== 제출 계약 검증: {BASE} ===")
c = httpx.Client(timeout=600.0, trust_env=False, follow_redirects=True)

# 1. /health
try:
    r = c.get(f"{BASE}/health")
    ok("GET /health = 200") if r.status_code == 200 else ng(f"GET /health = {r.status_code}")
except Exception as e:
    ng(f"GET /health 접속 실패: {e}")

# 2. /v1/models
mid = None
try:
    mid = c.get(f"{BASE}/v1/models").json()["data"][0]["id"]
    ok(f"GET /v1/models -> data[0].id = {mid}")
except Exception as e:
    ng(f"GET /v1/models 파싱 실패: {e}")

# 3. /v1/chat/completions — 평가 서버가 보내는 형태 그대로
long_essay = ("우리 사회는 인공지능 도입을 서둘러야 한다. 그 까닭은 다음과 같다. " * 30)
msg = ("[prompt_text]\n인공지능 도입에 대한 자신의 견해를 쓰시오.\n\n"
       f"[essay_text]\n{long_essay}")
body = {"model": mid or "kwriting-scorer",
        "messages": [{"role": "user", "content": msg}],
        "max_tokens": 2048, "temperature": 0.0, "top_p": 1.0,
        "seed": 42, "stop": ["Q:", "User:"]}
obj = None
try:
    res = c.post(f"{BASE}/v1/chat/completions", json=body).json()
    content = res["choices"][0]["message"]["content"]
    ok("OpenAI 호환 응답 구조 (choices[0].message.content)")
    if content.strip().startswith("```"):
        ng("마크다운 코드블록 사용 — FAQ상 금지, 파싱 실패로 0점 처리된다")
    else:
        ok("코드블록 없음")
    obj = json.loads(content)
    ok("content가 단일 JSON 객체로 파싱됨")
except Exception as e:
    ng(f"추론 응답 처리 실패: {e}")

# 4. 출력 형식
if obj is not None:
    for d in DIMS:
        if d not in obj:
            ng(f"{d} 누락"); continue
        if "score" not in obj[d] or "rationale" not in obj[d]:
            ng(f"{d}에 score 또는 rationale 누락"); continue
        s, rat = obj[d]["score"], obj[d]["rationale"]
        if not (1 <= s <= 5):
            ng(f"{d} 점수 범위 이탈: {s}")
        elif float(s) != int(s):
            ng(f"{d} 정수 아님: {s} — 반올림 규칙상 정수로 내보내야 한다")
        elif len(str(rat)) < 20:
            ng(f"{d} 근거가 너무 짧음: {rat!r}")
        else:
            ok(f"{d}: score={int(s)}, 근거 {len(str(rat))}자")
    ok("장문 입력(약 1500자) 정상 처리")

print()
if obj is not None:
    print(json.dumps(obj, ensure_ascii=False, indent=2)[:900])
print()
print(f"통과 {ok_n} / 실패 {fail_n}")
if fail_n == 0:
    print(">>> 계약 검증 통과. 제출 가능.")
else:
    print(">>> 실패 항목 있음. 제출하지 말 것.")
    sys.exit(1)
