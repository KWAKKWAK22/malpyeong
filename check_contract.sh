#!/usr/bin/env bash
# 제출 규정 12절 체크리스트를 그대로 검사한다.
# 사용: bash check_contract.sh [BASE_URL]   (기본 http://127.0.0.1:8000)
BASE="${1:-http://127.0.0.1:8000}"
C="curl -s --noproxy *"
PASS=0; FAIL=0
ok(){ echo "  PASS  $1"; PASS=$((PASS+1)); }
ng(){ echo "  FAIL  $1"; FAIL=$((FAIL+1)); }

echo "=== 제출 계약 검증: $BASE ==="

code=$($C -o /dev/null -w '%{http_code}' "$BASE/health")
[ "$code" = "200" ] && ok "GET /health = 200" || ng "GET /health = $code"

mid=$($C "$BASE/v1/models" | python3 -c 'import sys,json;print(json.load(sys.stdin)["data"][0]["id"])' 2>/dev/null)
[ -n "$mid" ] && ok "GET /v1/models -> data[0].id = $mid" || ng "GET /v1/models 파싱 실패"

# 실제 평가 서버가 보내는 형태 그대로 (stop, top_p, seed 포함)
LONG=$(python3 -c "print('우리 사회는 인공지능 도입을 서둘러야 한다. '*40)")
REQ=$(python3 - <<PY
import json
long = "우리 사회는 인공지능 도입을 서둘러야 한다. "*40
msg = "[prompt_text]\n인공지능 도입에 대한 자신의 견해를 쓰시오.\n\n[essay_text]\n" + long
print(json.dumps({"model":"$mid","messages":[{"role":"user","content":msg}],
 "max_tokens":2048,"temperature":0.0,"top_p":1.0,"seed":42,
 "stop":["Q:","User:"]}, ensure_ascii=False))
PY
)
RES=$($C -X POST "$BASE/v1/chat/completions" -H 'Content-Type: application/json' -d "$REQ")

python3 - <<PY
import json,sys
r = json.loads('''$RES''') if '''$RES'''.strip() else {}
try:
    c = r["choices"][0]["message"]["content"]
except Exception:
    print("  FAIL  choices[0].message.content 없음"); sys.exit(1)
print("  PASS  OpenAI 호환 응답 구조")
if c.strip().startswith("```"):
    print("  FAIL  마크다운 코드블록 사용 (FAQ상 금지)"); sys.exit(1)
try:
    o = json.loads(c)
except Exception as e:
    print("  FAIL  content가 JSON으로 파싱되지 않음:", e); print(c[:300]); sys.exit(1)
print("  PASS  content가 단일 JSON 객체")
for d in ("content","organization","expression"):
    if d not in o: print(f"  FAIL  {d} 누락"); sys.exit(1)
    if "score" not in o[d] or "rationale" not in o[d]:
        print(f"  FAIL  {d}에 score/rationale 누락"); sys.exit(1)
    s, rat = o[d]["score"], o[d]["rationale"]
    if not (1 <= s <= 5): print(f"  FAIL  {d} 점수 범위 이탈: {s}"); sys.exit(1)
    if float(s) != int(s): print(f"  FAIL  {d} 정수 아님: {s} (반올림 규칙상 정수여야 함)"); sys.exit(1)
    if len(rat) < 20: print(f"  FAIL  {d} 근거가 너무 짧음: {rat!r}"); sys.exit(1)
print("  PASS  3개 영역 x (정수 score + 근거) 완비")
print("  PASS  장문 입력 처리")
print()
print(json.dumps(o, ensure_ascii=False, indent=2)[:800])
PY
RC=$?
echo
if [ "$FAIL" -eq 0 ] && [ "$RC" -eq 0 ]; then
  echo ">>> 계약 검증 통과. 제출 가능."
else
  echo ">>> 실패 항목 있음. 제출하지 말 것."
  exit 1
fi
