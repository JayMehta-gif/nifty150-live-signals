"""Live prices from INDmoney's INDstocks API (https://api-docs.indstocks.com).

Read-only use: this module logs in, downloads the instrument list and asks for
quotes. It never calls an order endpoint.

Credentials come from environment variables (GitHub repository secrets):
    INDSTOCKS_CLIENT_ID    the Client ID shown after "Setup TOTP" on the INDstocks site
    INDSTOCKS_MPIN         your INDstocks MPIN
    INDSTOCKS_TOTP_SECRET  the authenticator secret shown once during TOTP setup
If any is missing, `available()` is False and callers fall back to Yahoo.

Notes from the docs: each successful /generate/token call cancels the previous
token, so one token is made per run; market-data endpoints need no static IP;
quotes take up to 1000 codes per call and one unknown code fails the whole call.
"""
import base64
import hashlib
import hmac
import json
import os
import struct
import time
import urllib.error
import urllib.parse
import urllib.request
from io import StringIO

BASE = "https://api.indstocks.com"
_TOKEN = None


def available():
    return all(os.environ.get(k) for k in ("INDSTOCKS_CLIENT_ID", "INDSTOCKS_MPIN", "INDSTOCKS_TOTP_SECRET"))


def totp(secret, at=None, step=30, digits=6):
    """RFC 6238 code from a base32 authenticator secret (what Google Authenticator shows)."""
    key = base64.b32decode(secret.replace(" ", "").upper() + "=" * (-len(secret.replace(" ", "")) % 8))
    counter = int((at if at is not None else time.time()) // step)
    mac = hmac.new(key, struct.pack(">Q", counter), hashlib.sha1).digest()
    off = mac[-1] & 0x0F
    code = (struct.unpack(">I", mac[off:off + 4])[0] & 0x7FFFFFFF) % 10 ** digits
    return str(code).zfill(digits)


def _request(path, params=None, body=None, headers=None, timeout=20):
    url = BASE + path + ("?" + urllib.parse.urlencode(params, safe=",") if params else "")
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method="POST" if body is not None else "GET",
                                 headers={"User-Agent": "technical-dashboard/1.0", **(headers or {})})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8")


def token():
    """One access token per run (cached). Raises on failure."""
    global _TOKEN
    if _TOKEN:
        return _TOKEN
    body = {"mpin": os.environ["INDSTOCKS_MPIN"], "totp": totp(os.environ["INDSTOCKS_TOTP_SECRET"])}
    out = json.loads(_request("/generate/token", body=body,
                              headers={"x-api-key": os.environ["INDSTOCKS_CLIENT_ID"], "Content-Type": "application/json"}))
    tok = out.get("token") or (out.get("data") or {}).get("token")
    if not tok:
        raise RuntimeError(f"no token in response: {str(out)[:200]}")
    _TOKEN = tok
    return tok


def instruments(source="equity"):
    """Instrument master CSV as text (equity / index / fno / mcx)."""
    return _request("/market/instruments", {"source": source}, headers={"Authorization": token()}, timeout=60)


def equity_codes(csv_text):
    """{NSE symbol: "NSE_<SECURITY_ID>"} for NSE cash-market EQ rows."""
    import csv
    out = {}
    for row in csv.DictReader(StringIO(csv_text)):
        if row.get("EXCH") != "NSE" or row.get("SERIES", "EQ") not in ("EQ", "BE", ""):
            continue
        sym = (row.get("SYMBOL_NAME") or row.get("TRADING_SYMBOL") or "").strip()
        sid = (row.get("SECURITY_ID") or "").strip()
        if sym and sid and sym not in out:
            out[sym] = f"NSE_{sid}"
    return out


def index_ids(csv_text):
    """{index name: SECURITY_ID} from the 3-column index file (read by position)."""
    import csv
    out = {}
    rows = list(csv.reader(StringIO(csv_text)))
    for r in rows[1:]:
        if len(r) >= 3 and r[0].strip() == "NSE":
            out[r[1].strip().upper()] = r[2].strip()
    return out


def quotes(codes, kind="full"):
    """{code: quote dict} for up to any number of codes, 1000 per call. A call
    that fails is skipped (one bad code fails a whole call, per the docs)."""
    out = {}
    codes = list(dict.fromkeys(codes))
    for i in range(0, len(codes), 1000):
        chunk = codes[i:i + 1000]
        try:
            res = json.loads(_request(f"/market/quotes/{kind}", {"scrip-codes": ",".join(chunk)},
                                      headers={"Authorization": token()}))
            out.update(res.get("data") or {})
        except urllib.error.HTTPError as e:
            print(f"[warn] INDstocks quotes: HTTP {e.code} {e.read()[:200]!r}")
        except Exception as e:  # noqa: BLE001
            print(f"[warn] INDstocks quotes: {e}")
        time.sleep(0.25)   # 5 requests a second is the documented limit
    return out
