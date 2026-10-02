"""Hand-built cases for the four reward rules, the parser and the abstain action.

    python test_selector_rft.py

The case that matters most is abstain-vs-malformed: both produce "no passage was
chosen", but one is a legal action worth 1 when nothing matches and the other is
a failure worth -1. Collapsing them would train the policy to break its own
output format whenever it wanted to abstain.
"""
import sys

import selector_rft as S

FAILED = []


def check(name, got, want):
    ok = got == want
    print("[%s] %-54s got %r" % ("ok  " if ok else "FAIL", name, got))
    if not ok:
        FAILED.append("%s: got %r want %r" % (name, got, want))


HAS = [0, 1, 0, 0, 1]        # n = 2: passages 2 and 5 carry a gold alias
NONE = [0, 0, 0, 0, 0]       # n = 0

print("\n== 1. 네 가지 보상 규칙 ==")
check("n>0, 정답 근거 선택 (2)", S.reward(2, HAS), 1.0)
check("n>0, 정답 근거 선택 (5)", S.reward(5, HAS), 1.0)
check("n>0, 잘못된 passage 선택", S.reward(1, HAS), 0.0)
check("n>0, 기권", S.reward(0, HAS), 0.0)
check("n=0, 기권", S.reward(0, NONE), 1.0)
check("n=0, passage 선택", S.reward(3, NONE), 0.0)
check("정답 정오는 보상에 없음 — 인자에 답이 없다",
      "answers" in S.reward.__code__.co_varnames, False)

print("\n== 2. 보고 버킷 ==")
check("n>0 적중", S.outcome(2, HAS), "n>0_hit")
check("n>0 오선택", S.outcome(1, HAS), "n>0_miss")
check("n>0 기권", S.outcome(0, HAS), "n>0_abstained")
check("n=0 기권", S.outcome(0, NONE), "n=0_abstained")
check("n=0 오선택", S.outcome(3, NONE), "n=0_picked")

print("\n== 3. 파서 — 기권과 형식 오류는 다른 것 ==")
check("정상 선택", S.parse_selection('{"source": 3}'), (3, True))
check("기권은 유효한 행동", S.parse_selection('{"source": 0}'), (0, True))
check("문자열 숫자", S.parse_selection('{"source": "4"}'), (4, True))
check("앞뒤 잡텍스트", S.parse_selection('Sure.\n{"source": 1}\n')[0], 1)
check("범위 밖 (6)", S.parse_selection('{"source": 6}'), (None, False))
check("음수", S.parse_selection('{"source": -1}'), (None, False))
check("불리언", S.parse_selection('{"source": true}'), (None, False))
check("키 없음", S.parse_selection('{"answer": "x"}'), (None, False))
check("JSON 깨짐", S.parse_selection('{"source":'), (None, False))
check("JSON 아님", S.parse_selection("passage 2"), (None, False))
check("빈 출력", S.parse_selection(""), (None, False))

print("\n== 4. 형식 오류 보상 ==")
check("기권 ≠ 형식 오류 (n=0)",
      (S.score_candidate('{"source": 0}', NONE)["reward"],
       S.score_candidate("I pick none", NONE)["reward"]), (1.0, -1.0))
check("형식 오류는 Bad 보다 낮다",
      S.score_candidate("nope", HAS)["reward"] < S.score_candidate('{"source": 1}', HAS)["reward"],
      True)
check("score_candidate 정상", S.score_candidate('{"source": 5}', HAS),
      {"text": '{"source": 5}', "valid": True, "source": 5, "reward": 1.0})

print("\n== 5. 라벨 ==")
docs = [{"title": "A", "text": "nothing here"},
        {"title": "Cyndi Lauper", "text": "sang True Colors"},
        {"title": "B", "text": "the answer is cyndi lauper indeed"}]
flags = S.passage_flags(docs, ["Cyndi Lauper"])
check("제목 일치도 센다", flags, [0, 1, 1])
check("n 계산", S.evidence_count(flags), 2)
check("관사 제거 후 일치", S.passage_flags([{"title": "", "text": "The Beatles"}],
                                      ["Beatles"]), [1])

print()
if FAILED:
    print("실패 %d건" % len(FAILED))
    for line in FAILED:
        print("  -", line)
    sys.exit(1)
print("전부 통과")
