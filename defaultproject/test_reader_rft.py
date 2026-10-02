"""Checks for the reward, the parser and the m computation. stdlib only.

    python test_reader_rft.py

The third check joins against the cached scores and reproduces `dig_order`'s
HasAnswer from scratch: if P(m > 0) does not land on the number already in
facts.json, the m used for the training filter is not the m the pipeline sees.
"""
import io
import json
import os
import sys

import reader_rft as R

FAILED = []


def check(name, got, want):
    ok = got == want
    print("[%s] %-56s got %r" % ("ok  " if ok else "FAIL", name, got))
    if not ok:
        FAILED.append("%s: got %r want %r" % (name, got, want))


def approx(name, got, want, tol):
    ok = abs(got - want) <= tol
    print("[%s] %-56s got %.4f want %.4f" % ("ok  " if ok else "FAIL", name, got, want))
    if not ok:
        FAILED.append("%s: got %.4f want %.4f" % (name, got, want))


# ------------------------------------------------------------------ 1. reward

print("\n== 1. 보상 순서 ==")
ANSWERS = ["Cyndi Lauper"]
# passage 1 holds the alias, passage 2 does not
FLAGS = [1, 0]

r_correct_hit = R.reward("Cyndi Lauper", 1, ANSWERS, FLAGS)
r_correct_miss = R.reward("Cyndi Lauper", 2, ANSWERS, FLAGS)
r_wrong_hit = R.reward("Zedd", 1, ANSWERS, FLAGS)
r_wrong_miss = R.reward("Zedd", 2, ANSWERS, FLAGS)

check("① 맞힘 + g=1", round(r_correct_hit, 6), 1.3)
check("② 맞힘 + g=0", round(r_correct_miss, 6), 0.5)
check("③ 틀림 + g=1", round(r_wrong_hit, 6), 0.0)
check("④ 틀림 + g=0", round(r_wrong_miss, 6), -0.5)
check("순서 ①>②>③>④",
      r_correct_hit > r_correct_miss > r_wrong_hit > r_wrong_miss, True)
# the constraint that fixes BETA < 1: a right answer with a bad citation must
# still beat a wrong answer with a good one, or the reward fights the metric
check("② > ③ (지표와 모순 없음)", r_correct_miss > r_wrong_hit, True)
check("정답 이득 ≥ 근거 이득",
      min(r_correct_hit - r_wrong_hit, r_correct_miss - r_wrong_miss)
      > max(r_correct_hit - r_correct_miss, r_wrong_hit - r_wrong_miss), True)

# ------------------------------------------------------------------ 2. parser

print("\n== 2. 파서 ==")
check("정상", R.parse_output('{"answer": "Cyndi Lauper", "source": 3}'),
      ("Cyndi Lauper", 3, True))
check("앞뒤 잡텍스트", R.parse_output('Sure!\n{"answer": "a", "source": 1}\nDone'),
      ("a", 1, True))
check("source 문자열", R.parse_output('{"answer": "a", "source": "2"}'), ("a", 2, True))
check("source 범위 밖", R.parse_output('{"answer": "a", "source": 9}'), ("a", None, False))
check("source 0", R.parse_output('{"answer": "a", "source": 0}'), ("a", None, False))
check("source 없음", R.parse_output('{"answer": "a"}'), ("a", None, False))
check("source 불리언", R.parse_output('{"answer": "a", "source": true}'), ("a", None, False))
check("JSON 깨짐", R.parse_output('{"answer": "a", "source":'), ("", None, False))
check("JSON 아님", R.parse_output("Cyndi Lauper"), ("", None, False))
check("빈 출력", R.parse_output(""), ("", None, False))
check("빈 answer 는 형식상 유효", R.parse_output('{"answer": "", "source": 1}'),
      ("", 1, True))
check("빈 answer 는 오답", R.is_correct("", ANSWERS), 0)

print("\n== 2b. 채점 ==")
check("부분문자열 포함", R.is_correct("the answer is Cyndi Lauper", ANSWERS), 1)
check("관사·대소문자 무시", R.is_correct("CYNDI LAUPER", ANSWERS), 1)
check("오답", R.is_correct("Zedd", ANSWERS), 0)

# --------------------------------------------------------------- 3. advantage

print("\n== 3. 어드밴티지 그룹 처리 ==")


def rec(text):
    return R.score_candidate(text, ANSWERS, FLAGS)


ok_hit = '{"answer": "Cyndi Lauper", "source": 1}'
ok_miss = '{"answer": "Cyndi Lauper", "source": 2}'
bad_hit = '{"answer": "Zedd", "source": 1}'
bad_miss = '{"answer": "Zedd", "source": 2}'
broken = "no json here"

_, why = R.group_advantage([rec(broken), rec(broken)])
check("형식 전멸", why, "no_valid")
_, why = R.group_advantage([rec(ok_hit), rec(ok_hit)])
check("영분산", why, "zero_variance")
# a wrong answer can sit above the group mean; e == 1 is what keeps it out
_, why = R.group_advantage([rec(bad_hit), rec(bad_miss), rec(bad_miss)])
check("정답 후보 없음 (오답이 평균 위여도 제외)", why, "no_correct_candidate")
w, why = R.group_advantage([rec(ok_hit), rec(bad_miss), rec(bad_miss)])
check("정상 그룹", why, None)
check("정상 그룹은 정답 후보만", sorted(w), [0])
approx("가중치 합 = 1", sum(w.values()), 1.0, 1e-9)
w, why = R.group_advantage([rec(ok_hit), rec(ok_miss), rec(bad_miss), rec(bad_miss)])
check("정답 2개면 둘 다 후보", sorted(w), [0, 1])
check("근거 적중 쪽 가중치가 더 큼", w[0] > w[1], True)
approx("가중치 합 = 1 (2개)", sum(w.values()), 1.0, 1e-9)

# ------------------------------------------------------- 4. m against facts

print("\n== 4. m 계산이 dig_order 의 HasAnswer 를 재현하는가 ==")
RES = "output/results"
needed = [os.path.join(RES, f) for f in
          ("bm25_top100.jsonl", "upr_scores.jsonl", "proxy_dig_scores.jsonl")]
if not all(os.path.exists(p) for p in needed):
    print("     건너뜀 — 캐시 없음")
else:
    sys.path.insert(0, ".")
    import analyze_position as A

    upr, proxy = {}, {}
    for line in io.open(needed[1], encoding="utf-8"):
        row = json.loads(line)
        if row.get("sig", "").startswith("upr|"):
            upr.setdefault(row["qid"], {}).update(row["scores"])
    for line in io.open(needed[2], encoding="utf-8"):
        row = json.loads(line)
        proxy.setdefault(row["qid"], {}).update(row["scores"])
    rows = A.load_run("dig_order")

    def rrf(score_lists, k=60):
        n = len(score_lists[0])
        fused = [0.0] * n
        for scores in score_lists:
            order = sorted(range(n), key=lambda i: scores[i], reverse=True)
            for rank, index in enumerate(order):
                fused[index] += 1.0 / (k + rank + 1)
        return fused

    present, total = 0, 0
    for line in io.open(needed[0], encoding="utf-8"):
        row = json.loads(line)
        qid = row["qid"]
        if qid not in upr or qid not in proxy or qid not in rows:
            continue
        pool = [p for p in row["passages"][:50] if p.get("text")]
        ids = [str(p.get("docid")) for p in pool]
        if not all(i in upr[qid] and i in proxy[qid] for i in ids):
            continue
        fused = rrf([[upr[qid][i] for i in ids], [proxy[qid][i] for i in ids]])
        top = sorted(range(len(pool)), key=lambda i: fused[i], reverse=True)[:5]
        answers = rows[qid][0]
        if not any(A.normalize_answer(a) for a in answers):
            continue
        total += 1
        present += int(R.answer_count([pool[i] for i in top], answers) > 0)
    approx("P(m>0) == facts.json dig_order HasAnswer", present / total, 0.6072, 0.0005)
    print("     n = %s" % format(total, ","))

print()
if FAILED:
    print("실패 %d건" % len(FAILED))
    for line in FAILED:
        print("  -", line)
    sys.exit(1)
print("전부 통과")
