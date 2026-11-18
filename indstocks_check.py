"""One-off INDmoney (INDstocks API) connection check, run by the
"INDmoney check" workflow. Prints what each step returns — never the secrets."""
import os
import sys
import time

import indstocks

print("secrets present:", {k: bool(os.environ.get(k)) for k in
                           ("INDSTOCKS_CLIENT_ID", "INDSTOCKS_MPIN", "INDSTOCKS_TOTP_SECRET")})
sec = os.environ.get("INDSTOCKS_TOTP_SECRET", "")
print("TOTP secret looks like base32:", bool(sec) and all(c.upper() in "ABCDEFGHIJKLMNOPQRSTUVWXYZ234567= " for c in sec),
      f"(length {len(sec.replace(' ', ''))})")
print("MPIN is digits only:", os.environ.get("INDSTOCKS_MPIN", "").strip().isdigit())
ok = True
try:
    indstocks.token()
    print("1. login (/generate/token): OK")
except Exception as e:  # noqa: BLE001
    print("1. login (/generate/token): FAILED —", e)
    sys.exit(1)
try:
    raw_eq = indstocks.instruments("equity")
    lines = raw_eq.splitlines()
    print("   equity CSV header:", lines[0][:400])
    for ln in [l for l in lines if "RELIANCE" in l.upper() or "HDFCBANK" in l.upper() or ",TCS" in l.upper()][:6]:
        print("   sample:", ln[:300])
    raw_ix = indstocks.instruments("index")
    print("   index CSV header:", raw_ix.splitlines()[0][:200])
    print("   NSE indices:", [l.split(",")[1] for l in raw_ix.splitlines()[1:] if l.startswith("NSE,")][:60])
    eq = indstocks.equity_codes(raw_eq)
    print(f"2. instruments (equity): OK — {len(eq)} NSE symbols, RELIANCE = {eq.get('RELIANCE')}")
except Exception as e:  # noqa: BLE001
    print("2. instruments: FAILED —", e)
    sys.exit(1)
try:
    idx = indstocks.index_ids(indstocks.instruments("index"))
    print(f"3. instruments (index): OK — NIFTY 50 = {idx.get('NIFTY 50')}, BANK NIFTY = {idx.get('BANK NIFTY')}, NIFTY 500 = {idx.get('NIFTY 500')}")
except Exception as e:  # noqa: BLE001
    print("3. index instruments: FAILED —", e)
    idx = {}
q = indstocks.quotes([eq["RELIANCE"]]) if eq.get("RELIANCE") else {}
print("4. live quote RELIANCE:", q or "FAILED (see warning above)")
ok &= bool(q)
now = int(time.time() * 1000)
try:
    c = indstocks.candles([eq["RELIANCE"]], "15minute", now - 4 * 86400000, now)
    rows = c.get(eq["RELIANCE"]) or []
    print(f"5. 15-min candles RELIANCE (4 days): {len(rows)} candles, last {rows[-1] if rows else None}")
    ok &= bool(rows)
except Exception as e:  # noqa: BLE001
    print("5. candles: FAILED —", e)
    ok = False
for prefix in ("NIDX_", "NSE_", "IDX_"):
    nid = idx.get("NIFTY 50")
    if not nid:
        break
    got = indstocks.quotes([f"{prefix}{nid}"])
    print(f"6. NIFTY 50 quote with prefix {prefix}: {'OK ' + str(got) if got else 'no'}")
    if got:
        break
sys.exit(0 if ok else 1)
