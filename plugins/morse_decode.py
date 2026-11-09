#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""plugins/morse_decode.py — 示例插件内核：摩斯码解码。

插件协议（见 plugins/README.md）：
  1. 模块级必须提供 `KERNEL = dict(name=..., help=...)`；
  2. 必须能被 `python plugins/morse_decode.py <args>` 直接调用，结果写 stdout，成功返回 0；
  3. 不要 import 仓库根模块（插件应自包含），也不要和内置内核重名。
"""
from __future__ import annotations

import argparse
import sys

KERNEL = dict(
    name='morse',
    help='摩斯码解码（支持 / 分词、多种分隔符）',
)

MORSE = {
    '.-': 'A', '-...': 'B', '-.-.': 'C', '-..': 'D', '.': 'E', '..-.': 'F',
    '--.': 'G', '....': 'H', '..': 'I', '.---': 'J', '-.-': 'K', '.-..': 'L',
    '--': 'M', '-.': 'N', '---': 'O', '.--.': 'P', '--.-': 'Q', '.-.': 'R',
    '...': 'S', '-': 'T', '..-': 'U', '...-': 'V', '.--': 'W', '-..-': 'X',
    '-.--': 'Y', '--..': 'Z',
    '-----': '0', '.----': '1', '..---': '2', '...--': '3', '....-': '4',
    '.....': '5', '-....': '6', '--...': '7', '---..': '8', '----.': '9',
    '.-.-.-': '.', '--..--': ',', '..--..': '?', '-..-.': '/', '-....-': '-',
    '-.--.': '(', '-.--.-': ')', '.-..-.': '"', '.----.': "'", '---...': ':',
    '-.-.-.': ';', '..--.-': '_', '.-...': '&', '.-.-.': '+', '-...-': '=',
    '.--.-.': '@',
}


def decode(code, word_sep='/', letter_sep=None):
    """把摩斯串解码为文本。letter_sep 为空时按空白分词，否则按给定字符分词。"""
    tokens = code.replace('|', '/').split(word_sep)
    words = []
    for tok in tokens:
        tok = tok.strip()
        if not tok:
            words.append('')
            continue
        parts = tok.split(letter_sep) if letter_sep else tok.split()
        out = []
        for p in parts:
            if not p:
                continue
            ch = MORSE.get(p)
            out.append(ch if ch else f'?{p}?')
        words.append(''.join(out))
    return ' '.join(x for x in words if x != '') if not letter_sep else ' '.join(words)


def main(argv=None):
    ap = argparse.ArgumentParser(description='摩斯码解码')
    ap.add_argument('--code', help='摩斯串，例如 "... --- ... / .-- --- .-. .-.. -.."')
    ap.add_argument('--file', help='从文件读取摩斯串')
    ap.add_argument('--letter-sep', default=None,
                    help="字母分隔符（默认按空白分词；密集串可用 '' 配合 / 分词）")
    a = ap.parse_args(argv)
    if a.file:
        with open(a.file, encoding='utf-8') as fh:
            code = fh.read()
    elif a.code:
        code = a.code
    else:
        print('[错误] 需要 --code 或 --file', file=sys.stderr)
        return 2
    text = decode(code, letter_sep=a.letter_sep)
    print(f'[morse] {text}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
