---
name: ctf-toolkit
description: GPU-accelerated, command-line CTF toolkit for batch brute-force and triage (MT19937 seed scan, AES S-box fault scan, ChaCha20/RC4/AES-CTR keystream, RS(63,31) syndrome, TEA/XTEA/XXTEA/Speck, LCG recovery, MD5 cracking, RSA multi-attack, multi-layer decoding, image LSB/stegscan, image diff, ROP gadgets, ELF recon, cyclic offsets). Use when a CTF task requires trying many candidates at once, identifying an unknown cipher, recovering keys or keystreams, extracting hidden data from images or files, or computing pwn offsets and gadgets.
---

# CTF Toolkit

A Python toolkit covering four families of CTF work — crypto kernels, forensics/stego, pwn helpers, and
GPU batch attackers. It exists because serial Python loops waste hours on work a GPU finishes in
seconds: always reach for a batch kernel before writing a new brute-force script.

## 1. Locate the toolkit and the interpreter

The dispatcher (`scripts/ctf.py`, shipped with this skill) finds the toolkit directory on its own:

1. `CTF_TOOLKIT_DIR` if set and it actually contains `gpu_crack.py`;
2. otherwise it walks up from its own location looking for `gpu_crack.py` — so a skill directory nested
   anywhere inside the repository works with no configuration;
3. otherwise test gear fails loudly with exit code 2 and an explicit message.

Set `CTF_TOOLKIT_DIR` to the repository root **only when** this skill directory has been copied outside
the repository (e.g. into `~/.codex/skills/ctf-toolkit`). That is the one deployment case the upward scan
cannot resolve.

Interpreter selection is fully automatic (no hard-coded paths), in this order: `CTF_PYTHON` (honoured
unconditionally — only the file has to exist, CUDA is not checked) → the current interpreter →
`python3`/`python` on `PATH` → Windows `py -0p` entries → a `.venv`/`venv` inside the toolkit directory.
Every candidate except the first must pass `import torch; torch.cuda.is_available()`; the first one that
does wins. If none qualifies, the current interpreter is used and GPU kernels fall back to the slow
`--backend numpy` path instead of failing. Set `CTF_PYTHON` to pin a specific environment.

## 2. Triage the task, then run one kernel

| If the task is… | Run |
|---|---|
| unknown stream cipher, many candidate keys | `mtscan` (mt19937 seed), `sboxscan`, `ksscore`, `lfsrscan` |
| ChaCha20 keystream, Poly1305 tag, RC4, AES-CTR | `chacha`, `poly1305`, `rc4`, `aesctr` |
| AES-Sbox fault analysis | `sboxscan` (128/192/256-bit keys) |
| Reed-Solomon / error-correcting code over GF(256) | `rs63` |
| repeating-XOR or single-byte XOR ciphertext | `xorbrute` |
| TEA/XTEA/XXTEA/Speck encrypted blob | `tea`, `speck` |
| pseudo-random outputs to predict | `lcg`, then `mtscan` |
| MD5 hash to crack | `md5` |
| forge a MAC/signature over an extended message | `mdext` (MD5 / SHA-1 length extension) |
| output bitstream, unknown LFSR taps | `bm` (Berlekamp-Massey) |
| encrypted ZIP | `zipbrute` (wordlist or mask) |
| ECC public key with a small private key | `eccscan` (secp256k1 / P-256, ~1.1 M keys/s per core) |
| RSA modulus + ciphertext, unknown attack | `rsa` (tries five attacks in cost order) |
| unknown encoding layers | `encode`, `crc32` |
| image with suspected LSB stego | `stegscan` then `lsb` |
| file with appended/embedded data | `stegscan` |
| two images that differ | `imgdiff` |
| ELF protections, imports, PLT/GOT | `elfinfo` |
| ROP chain needed | `gadgets` |
| overflow offset needed | `cyclic` + `cycfind` |

Full kernel list with throughput numbers and worked examples: `references/kernels.md`.
The kernel table's single source of truth is `ctf_registry.py`; extra kernels can be dropped into
`plugins/*.py` with a module-level `KERNEL = dict(name=..., help=...)` — no repo edits required.
For any command's own flags: `python scripts/ctf.py <module> --help` (the runtime source of truth —
never guess parameter names).

## 3. Run it

Use the bundled dispatcher so path and interpreter selection stay correct:

```
python scripts/ctf.py <module> <args...>
python scripts/ctf.py selftest        # 4-module combo self-test; every kernel has a known-answer vector
```

Examples:

```
python scripts/ctf.py rsa --n <hex> --e 65537 --c <hex>
python scripts/ctf.py mtscan --key-hex ac2f75c043fbc36709d315f2 --seed-start 0 --seed-count 1000000
python scripts/ctf.py lsb --img chal.png
python scripts/ctf.py gadgets --file vuln --filter "syscall"
```

Global options work before or after the module name: `--backend auto|torch|numpy`, `--chunk N`
(OOM halves it automatically), `--json` (one-line JSON result), `--quiet`, `--no-progress`,
`--checkpoint FILE` / `--reset-checkpoint` (resumable scans). `ctf_misc.py` is pure CPU and
accepts none of them.

## 4. Output contract (parse only these)

| Signal | Meaning |
|---|---|
| exit code `0` | success — including "no hit"/"not cracked"; read the output, not just the status |
| exit code `1` | self-test failed, or a verification mismatch |
| exit code `2` | bad arguments/input, unknown module, **or toolkit directory not found** |
| `★` in a line | hit / candidate / suspicious line; hex fields are lower-case, without `0x` |
| `PASS` / `FAIL` | self-test verdicts |
| stderr | errors; Chinese wording may change between versions — never match on the text |

When output is piped or redirected, the dispatcher forces UTF-8, so decode stdout as UTF-8. On an
interactive Windows console it keeps the system code page instead.

## 5. Before trusting a result

- Run `python scripts/ctf.py selftest` before and after touching any kernel.
- Run `consistency.py` from the repository root after editing crypto kernels: it cross-checks the same
  answer through torch, numpy, pure-Python references, pycryptodome and `cryptography` (45 assertions).
- Run `python -X utf8 fuzz_kernels.py --quick` for randomized differential testing against independent
  reference implementations (17 kernel categories; `verify_all.py` runs every layer in one command).
- Prefer the cheap predicate first: compare a 4-byte prefix or run XOR+popcount scoring, and only then
  run the full check (Poly1305 tag, complete AES block). This ordering is the single biggest speedup.
- Re-verify every hit through a **second, independent path** (decrypt with the recovered key and compare;
  recompute the LSB bit stream from raw pixels). A single hit is not a conclusion.
- Treat a "hit" containing text from another challenge's writeup, a sample prompt, or a previous task as
  a false positive — verify the source before reporting it.
- When a search returns the same best score for hundreds of candidates with no improvement, stop and
  change the model assumption instead of widening the search.

## 6. Keep scans bounded

Batch kernels will happily run for hours. Before any large run:

1. probe small (`--seed-count 1000`, `--tries 1000`, `--max-len 1`, a short ciphertext slice) and confirm
   the flags, the output shape and the exit code;
2. read throughput from `python scripts/ctf.py bench --help` / `bench` output;
3. size the real run from that number, keep `--chunk` modest on 8 GB VRAM cards (halve it on OOM), and
   split multi-billion candidate spaces into stages you can resume.

## 7. Boundaries

- Never submit a recovered flag to a competition platform; report it and let the operator decide.
- Do not read browser profiles, credentials, session stores, or other private user data while solving.
- Do not run unbounded searches: always give `--seed-count`, `--tries` or `--chunk` an explicit ceiling.
