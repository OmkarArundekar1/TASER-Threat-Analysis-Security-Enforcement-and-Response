import collections
import json

def view(n):
    return json.load(open(f"runs/{n}/view.json"))

for n in ("CUR_norestart", "A_norestart", "B_norestart"):
    v = view(n)
    print("=====", n, "| view keys", sorted(v))
    print("campaign keys", sorted(v["campaigns"][0]))
    print("sample transition", (v.get("transitions") or [None])[0] if isinstance(v.get("transitions"), list) else v.get("transitions"))
    by_dec = collections.defaultdict(list)
    for d in v["lifecycle_decisions"]:
        if d.get("score") is not None:
            by_dec[d["decision"]].append(d["score"])
    for k, s in sorted(by_dec.items()):
        s = sorted(s)
        print(f"  {k}: n={len(s)} min={s[0]:.3f} max={s[-1]:.3f} above1={sum(x > 1 for x in s)}")
    pairs = collections.Counter()
    total = collections.Counter()
    for c in v["campaigns"]:
        self_loop = c["attacker_ip"] == c["victim_ip"]
        seq = [t for t in c["sequence"] if t and t != "UNKNOWN"]
        total["self" if self_loop else "network"] += 1
        if seq:
            total["self_with_technique" if self_loop else "network_with_technique"] += 1
        for a, b in zip(seq, seq[1:]):
            if a != b:
                pairs["self" if self_loop else "network"] += 1
    print("  campaigns", dict(total))
    print("  adjacent distinct technique pairs in sequences, by identity", dict(pairs))
