#!/usr/bin/python3

# used for transform latex formulas into an image url, which can be displayed on GitHub
# usage: cat origin.md | trans.py > out.md
# google api: https://developers.google.com/chart/infographics/docs/formulas

import fileinput
import re
from urllib.parse import urlencode


def formula_to_img(match):
  formula = match.group(1).replace('\n', '')
  q = {
      'cht': 'tx',
      'chf': 'bg,s,00000000',
      'chl': r'\Large ' + formula
  }
  return '![](https://chart.googleapis.com/chart?%s)' % urlencode(q)


lines = []
for line in fileinput.input():
  lines.append(line)
content = ''.join(lines)

content = re.sub(
    r'(?:\\begin\{.+?\})|(?:\\end\{.+?\})|(?:\\\\)',
    '',
    content
)
content = re.sub(
    r'&=',
    '=',
    content
)

content = re.sub(
    r'\$\$(.+?)\$\$',
    formula_to_img,
    content,
    flags=re.DOTALL
)
content = re.sub(
    r'\$(.+?)\$',
    formula_to_img,
    content,
)

print(content)
