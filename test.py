from ordinal import Ord, FGH

latex_html_headers = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8">
  <title>FGH </title>
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css">
  <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.js"></script>
  <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/contrib/auto-render.min.js"
    onload="renderMathInElement(document.body, { delimiters: [
      {left: '\\\\(', right: '\\\\)', display: false},
      {left: '$$', right: '$$', display: true}
  ] });">
  </script>
  <style>
    body {
      font-family: sans-serif;
      padding: 2em;
    }
  </style>
</head>
<body>
"""

latex_html_ends = r"""</body>
</html>
"""

def latex_to_html(latex_str_list, path):
  with open(path, "w") as file:
    file.write(latex_html_headers)
    file.write('\n')
    for s in latex_str_list:
      file.write(f'<p>$$ {s} $$</p>\n')
    file.write('\n')
    file.write(latex_html_ends)

if __name__ == '__main__':
  # Example: w * 2 + 3
  expr_tree = Ord('+',
                   Ord('*', Ord('w'), Ord('2')),
                   Ord('3'),
                  )
  print(expr_tree)  # Output: Infix Expression: (3 + (w * 2))

  f223 = 2**24*24
  latex_to_html([
    expr_tree.to_latex(),
    Ord('1').fundamental_sequence_display(3, Ord('1')),
    Ord('w').fundamental_sequence_display(3, Ord('3')),
    Ord.from_str('w+2').fundamental_sequence_display(3, Ord.from_str('w+2')),
    Ord.from_str('w^2+w').fundamental_sequence_display(3, Ord.from_str('w^2+3')),
    Ord.from_str('w*1').fundamental_sequence_display(3, Ord.from_str('3')),
    Ord.from_str('w*2').fundamental_sequence_display(3, Ord.from_str('w+3')),
    Ord.from_str('w*w').fundamental_sequence_display(4, Ord.from_str('w*3+4')),
    Ord.from_str('w*(w+1)').fundamental_sequence_display(3, Ord.from_str('w*w+3')),
    Ord.from_str('w^1').fundamental_sequence_display(4, Ord.from_str('4')),
    Ord.from_str('w^2').fundamental_sequence_display(3, Ord.from_str('w*2+3')),
    Ord.from_str('w^w').fundamental_sequence_display(3, Ord.from_str('w^2*2+(w*2+3)')),
    Ord.from_str('w^(w+1)').fundamental_sequence_display(3,
      Ord.from_str('(((w ^ w) * 2) + (((w ^ 2) * 2) + ((w * 2) + 3)))')),
    Ord.from_str('e').fundamental_sequence_display(3),
    Ord.from_str('e*w').fundamental_sequence_display(3),
    Ord.from_str('e^w').fundamental_sequence_display(3),
    # FGH
    FGH(Ord.from_str('w^w'), 3).to_latex(),
    FGH(Ord.from_str('w^w'), FGH(Ord.from_str('w^w'), 3)).to_latex(),
    FGH(Ord.from_str('w^w'), 3).expand_once_display(FGH(Ord.from_str('w^2*2+(w*2+3)'), 3)),
    FGH(Ord('3'), 3).expand_once_display(FGH.seq('2', '2', '2', 3)),
    FGH(Ord('w'), 3).expand_once_display(FGH(Ord('3'), 3)),
    FGH(Ord('0'), 3).expand_once_display(4),
    FGH(Ord('1'), 3).expand_once_display(6),
    FGH(Ord('2'), 3).expand_display(24),
    FGH(Ord('3'), 3).expand_display(FGH(Ord('2'), f223)),
    FGH(Ord.from_str('w+1'), 3).expand_display(FGH.seq('w', 'w', '2', f223)),
    # FGH(Node('w^w'), 3).expand_display(),
    FGH(Ord.from_str('w^w+1'), 3).expand_display(),
    FGH(Ord.from_str('w^(w^w)'), 2).expand_display(),  # todo: correct?
    FGH(Ord.from_str('w^(w^w)'), 3).expand_display(limit=3),  # todo: smarter length ctrl
  ], './test.html')
