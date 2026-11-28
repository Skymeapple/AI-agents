"""Stricter provenance count: only data-derived targets that name an external party
(recipient/participant/email/IBAN/url/user), excluding internal ids and amounts."""
import csv, json, os, collections
RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
EXT = {"recipients", "participants", "recipient", "email", "user_email", "url", "user"}
c = collections.Counter(); tot = collections.Counter()
for r in csv.DictReader(open(os.path.join(RES, "target_provenance.csv"))):
    if r["consequential"] == "1":
        tot[r["suite"]] += 1
    args = [a for a in r["derived_args"].split("; ") if a]
    if any(a.split("=")[0].split(".")[1] in EXT for a in args):
        c[r["suite"]] += 1
out = {"external_party_data_derived": dict(c), "total": sum(c.values()), "consequential": dict(tot)}
json.dump(out, open(os.path.join(RES, "provenance_strict.json"), "w"), indent=1); print(out)
