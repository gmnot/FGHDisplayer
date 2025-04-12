from enum import Enum
from ordinal import Ord, FGH

latex_html_headers = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8">
  <title>Test</title>
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css">
  <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.js"></script>
  <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/contrib/auto-render.min.js"
    onload="renderMathInElement(document.body, { delimiters: [
      {left: '$$', right: '$$', display: true},
      {left: '$', right: '$', display: false}
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

class OutType(Enum):
  DIV    = 1
  PLAIN  = 2

def latex_to_html(latex_str_list, path):
  with open(path, "w") as file:
    file.write(latex_html_headers)
    file.write('\n')
    for s in latex_str_list:
      if isinstance(s, str):
        file.write(f'<p>$$ {s} $$</p>\n')
      else:
        match s[0]:
          case OutType.PLAIN:
            file.write(f'{s[1]}')
          case OutType.DIV:
            file.write(f'<div>$$ {s[1]} $$</p>\n')
          case _:
            assert 0, s[0]
    file.write('\n')
    file.write(latex_html_ends)

def test_f_s(ord1 : str, n : int, ord2=None, test_only=False, show_steps=False):
  formula = Ord.from_str(ord1).fundamental_sequence_display(
            n,
            expected=Ord.from_str(ord2) if ord2 is not None else None,
            test_only=test_only,
            show_steps=show_steps)
  if show_steps:
    return (OutType.DIV, formula)
  else:
    return formula

if __name__ == '__main__':
  # Example: w * 2 + 3
  expr_tree = Ord('+',
                   Ord('*', Ord('w'), Ord('2')),
                   Ord('3'),
                  )
  print(expr_tree)  # Output: Infix Expression: (3 + (w * 2))

  f223 = 2**24*24
  f2_256 = 29642774844752946028434172162224104410437116074403984394101141506025761187823616

  tests = [
    expr_tree.to_latex(),
    # FGH(Ord.from_str('w^w'), 3).to_latex(),
    # FGH(Ord.from_str('w^w'), FGH(Ord.from_str('w^w'), 3)).to_latex(),
    (OutType.PLAIN, r'<h3> $ f_c(n) $ </h3>'+'\n'),
    FGH(0, 3  ).expand_display(4),
    FGH(1, 3  ).expand_display(6),
    FGH(2, 3  ).expand_display(24),
    FGH(2, 256).expand_display(f2_256),
    FGH(3, 3  ).expand_display(FGH(2, f223)),

    (OutType.PLAIN, r'<h3> $ \omega^\alpha $ </h3>'+'\n'),
    test_f_s('1'  , 3, '1', test_only=True),
    test_f_s('w'  , 3, '3', test_only=True),
    test_f_s('w*1', 3, '3', test_only=True),
    test_f_s('w^1', 4, '4', test_only=True),
    FGH('w', 3).expand_display(FGH(2, f223)),
    FGH(Ord.from_str('w+1'), 3).expand_display(FGH('w', FGH(2, f223), 2)),
    test_f_s('w+2'                , 3, 'w+2'   , test_only=True),
    test_f_s('w+w'                , 4, 'w+4'),
    test_f_s('w*2'                , 3, 'w+3'),
    test_f_s('w*w'                , 4, 'w*3+4', test_only=True),
    test_f_s('w^2'                , 3, 'w*2+3', test_only=True),
    test_f_s('w^2'                , 4, 'w*3+4'),
    test_f_s('w*(w+1)'            , 3, 'w*w+3', test_only=True),
    test_f_s('w^2+w'              , 3, 'w^2+3'),
    test_f_s('w^3'                , 4, 'w^2*3+(w*3+4)'),
    test_f_s('w^w'                , 3, 'w^2*2+(w*2+3)'),
    test_f_s('w^(w+1)'            , 3, '(w^w)*2 + ((w^2)*2 + (w*2+3))', show_steps=True),
    test_f_s('w^((w^2)*2+(w*2)+2)', 3, show_steps=True),

    (OutType.PLAIN, r'<h3> $ \varepsilon_0 $ </h3>'+'\n'),
    test_f_s('e'                  , 3, show_steps=True),
    test_f_s('e*w'                , 3, show_steps=True),
    test_f_s('e^w'                , 3, show_steps=True),

    (OutType.PLAIN, '<h3>FGH</h3>\n'),
    FGH('w^w', 3).expand_once_display(FGH(Ord.from_str('w^2*2+(w*2+3)'), 3)),
    FGH('w^w', 3).expand_display(show_steps=True),
    FGH(Ord.from_str('w^w+1'), 3).expand_display(show_steps=True),
    FGH(Ord.from_str('w^(w^w)'), 2).expand_display(show_steps=True),  # todo: correct?
    # todo: smarter length ctrl
    FGH(Ord.from_str('w^(w^w)'), 3).expand_display(limit=3, show_steps=True),
  ]

  latex_to_html([s for s in tests if s is not None], './test.html')
