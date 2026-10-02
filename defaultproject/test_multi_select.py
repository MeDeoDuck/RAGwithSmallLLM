"""Reward and parser checks for multi-passage selection.

    python test_multi_select.py

The seven T = {2, 4} cases are the ones specified for this experiment; the rest
cover n = 0, the n = 1 and n = 5 edges, and every way an output can be malformed.
The case worth staring at is [2] = 2/3 against [2, 4] = 1: the reward pays for
listing both answer-bearing passages even when one of them is redundant.
"""
import sys

import multi_select_rft as M

FAILED = []


def check(name, got, want):
    ok = got == want
    print("[%s] %-50s got %r" % ("ok  " if ok else "FAIL", name, got))
    if not ok:
        FAILED.append("%s: got %r want %r" % (name, got, want))


def close(name, got, want, tol=1e-9):
    ok = abs(got - want) <= tol
    print("[%s] %-50s got %.4f want %.4f" % ("ok  " if ok else "FAIL", name, got, want))
    if not ok:
        FAILED.append("%s: got %.4f want %.4f" % (name, got, want))


# T = {2, 4}: passages 2 and 4 carry a gold alias
T24 = [0, 1, 0, 1, 0]

print("\n== 1. 지정된 보상 사례  T={2,4}, n=2 ==")
close("[2,4]      완전 일치", M.reward([2, 4], T24), 1.0)
close("[2]        절반만", M.reward([2], T24), 2 / 3)
close("[2,3,4]    둘 다 + 군더더기 1", M.reward([2, 3, 4], T24), 4 / 5)
close("[2,3]      하나 맞고 하나 틀림", M.reward([2, 3], T24), 1 / 2)
close("[1,2,3,4,5] 전부 선택", M.reward([1, 2, 3, 4, 5], T24), 4 / 7)
close("[1,3]      전부 오선택", M.reward([1, 3], T24), 0.0)
close("[]         빈 목록", M.reward([], T24), 0.0)

print("\n== 2. n=0 ==")
NONE = [0, 0, 0, 0, 0]
close("n=0, []  기권 성공", M.reward([], NONE), 1.0)
close("n=0, [3] 오선택", M.reward([3], NONE), 0.0)
close("n=0, [1,2,3,4,5]", M.reward([1, 2, 3, 4, 5], NONE), 0.0)

print("\n== 3. 경계 n=1, n=5 ==")
T1 = [0, 0, 1, 0, 0]
close("n=1, [3]   정확", M.reward([3], T1), 1.0)
close("n=1, [3,4] 군더더기 1", M.reward([3, 4], T1), 2 / 3)
close("n=1, [4]   오선택", M.reward([4], T1), 0.0)
close("n=1, []", M.reward([], T1), 0.0)
T5 = [1, 1, 1, 1, 1]
close("n=5, 전부 선택", M.reward([1, 2, 3, 4, 5], T5), 1.0)
close("n=5, [1]만", M.reward([1], T5), 2 / 6)
close("n=5, []", M.reward([], T5), 0.0)

print("\n== 4. 파서 — 빈 목록은 유효, 정렬 위반은 오류 ==")
check("정상", M.parse_sources('{"sources": [2, 4]}'), ([2, 4], True))
check("빈 목록은 유효한 선택", M.parse_sources('{"sources": []}'), ([], True))
check("단일", M.parse_sources('{"sources": [5]}'), ([5], True))
check("앞뒤 잡텍스트", M.parse_sources('Here:\n{"sources": [1, 2]}\n'), ([1, 2], True))
check("정렬 위반", M.parse_sources('{"sources": [4, 2]}'), (None, False))
check("중복", M.parse_sources('{"sources": [2, 2]}'), (None, False))
check("범위 밖 0", M.parse_sources('{"sources": [0]}'), (None, False))
check("범위 밖 6", M.parse_sources('{"sources": [6]}'), (None, False))
check("음수", M.parse_sources('{"sources": [-1]}'), (None, False))
check("정수 아님", M.parse_sources('{"sources": [1.5]}'), (None, False))
check("문자열 원소", M.parse_sources('{"sources": ["2"]}'), (None, False))
check("불리언 원소", M.parse_sources('{"sources": [true]}'), (None, False))
check("길이 초과", M.parse_sources('{"sources": [1,2,3,4,5,5]}'), (None, False))
check("리스트 아님", M.parse_sources('{"sources": 2}'), (None, False))
check("키 다름", M.parse_sources('{"source": [2]}'), (None, False))
check("JSON 깨짐", M.parse_sources('{"sources": [2'), (None, False))
check("JSON 아님", M.parse_sources("passages 2 and 4"), (None, False))
check("빈 출력", M.parse_sources(""), (None, False))

print("\n== 5. 형식 오류 보상 ==")
check("형식 오류 = −1", M.score_candidate("nope", T24)["reward"], -1.0)
check("형식 오류 ≠ 올바른 빈 목록",
      (M.score_candidate('{"sources": []}', NONE)["reward"],
       M.score_candidate("none of them", NONE)["reward"]), (1.0, -1.0))
check("정렬 위반도 −1", M.score_candidate('{"sources": [4,2]}', T24)["reward"], -1.0)
check("score_candidate 정상",
      M.score_candidate('{"sources": [2, 4]}', T24),
      {"text": '{"sources": [2, 4]}', "valid": True, "sources": [2, 4],
       "reward": 1.0, "k": 2, "tp": 2})

print("\n== 6. 보고용 지표 ==")
s = M.selection_scores([2, 3], T24)
check("n>0 precision", s["precision"], 0.5)
check("n>0 recall", s["recall"], 0.5)
close("n>0 f1", s["f1"], 0.5)
check("완전 일치 아님", s["exact"], False)
check("완전 일치", M.selection_scores([2, 4], T24)["exact"], True)
e = M.selection_scores([], T24)
check("n>0 빈 목록: precision 은 정의 안 함", e["precision"], None)
check("n>0 빈 목록: recall 0", e["recall"], 0.0)
close("n>0 빈 목록: f1 0", e["f1"], 0.0)
z = M.selection_scores([], NONE)
check("n=0 기권: f1 은 정의 안 함", z["f1"], None)
check("n=0 기권 성공", z["abstain_correct"], True)
check("n=0 오선택", M.selection_scores([2], NONE)["abstain_correct"], False)

print("\n== 7. 보상이 '전부 선택'을 요구한다는 사실 ==")
check("중복 근거라도 둘 다 골라야 만점",
      M.reward([2, 4], T24) > M.reward([2], T24), True)
check("최소 충분 근거에 보너스 없음",
      M.reward([2], T24) < 1.0, True)

print()
if FAILED:
    print("실패 %d건" % len(FAILED))
    for line in FAILED:
        print("  -", line)
    sys.exit(1)
print("전부 통과")
