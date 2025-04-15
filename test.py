from enum import Enum
import ordinal
from ordinal import Ord, FdmtSeq, FGH, get_rotate_counter, WIPError, ord_set_debug_mode
import utils

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
  print(f'update: {utils.get_file_mtime_str(path)}')

def test_display(obj, expected=None, *,
                 limit=None, until=None, test_only=False,
                 show_step=False, print_str=False):

  recorder = ordinal.FSRecorder((15 if show_step else 1),
                                limit if limit else obj.cal_limit_default,
                                until=until)
  res = obj.calc(recorder)

  if expected is not None:
    assert res == expected, f'{res} != {expected}'

  if test_only:
    return None

  if not show_step:
    return f'{ordinal.to_latex(obj)}={ordinal.to_latex(res)}'

  if print_str:
    print(res)

  assert recorder is not None

  formula = recorder.to_latex(ordinal.to_latex)
  if show_step:
    return (OutType.DIV, formula)
  else:
    return formula

def test_f_s(ord1 : str | int, n : int, expected=None, **kwargs):

  return test_display(
    FdmtSeq(ord1, n, latex_force_veblen=True),
    expected, **kwargs
  )

def test_fgh(ord1 : str | int, n : int, expected=None, **kwargs):

  return test_display(
    FGH(ord1, n, latex_force_veblen=True),
    expected, **kwargs
  )


def test_associative():
  for s1, s2 in [('(w*2+w)+1', 'w*2+(w+1)'),
                 ('((w^2+w*2)+w)+1', 'w^2+(w*2+(w+1))'),
                 ('(((w^w+w^2)+w*2)+w)+1', 'w^w+(w^2+(w*2+(w+1)))'),
                 ('(w^2*w)*2', 'w^2*(w*2)'),
                 ]:
    ord1, ord2 = Ord.from_any(s1), Ord.from_any(s2)
    assert ord1 == ord2, f'\n{ord1}\n{ord2}'

def test_exceptions():
  cnt = 0
  cases = [('v(2,0,0)', WIPError)]
  for ord_str, Err in cases:
    try:
      test_f_s(Ord.from_any(ord_str), 3)
    except Err:
      cnt += 1
    except Exception as e:
      raise f'Exception of unexpected type {e}'
  print(f'Test Exceptions: {cnt}/{len(cases)} cases passed')

@utils.track_total_time()
def test_main():
  ord_set_debug_mode(True)
  # Example: w * 2 + 3
  expr_tree = Ord('+',
                   Ord('*', Ord('w'), Ord('2')),
                   Ord('3'),
                  )
  print(expr_tree)  # Output: Infix Expression: (3 + (w * 2))

  f223 = 2**24*24
  f2_256 = 29642774844752946028434172162224104410437116074403984394101141506025761187823616

  test_associative()
  # test_exceptions()

  tests = [
    expr_tree.to_latex(),
    (OutType.PLAIN, r'<h2> $ f_c(n) $ </h2>'+'\n'),
    test_fgh(0,     3, 4),
    test_fgh(1, 3  , 6),
    test_fgh(2, 3  , 24),
    test_fgh(2, 256, f2_256, test_only=True),
    test_fgh(3, 2  , 2048),
    test_fgh(3, 3  , FGH(2, f223)),

    (OutType.PLAIN, r'<h2> $ \omega^\alpha $ </h2>'+'\n'),
    test_f_s( 1   , 3, 1, test_only=True),
    test_f_s('w'  , 3, 3, test_only=True),
    test_f_s('w*1', 3, 3, test_only=True),
    test_f_s('w^1', 4, 4, test_only=True),
    test_fgh('w'  , 3, FGH(2, f223)),
    # test_fgh('1+w', 3, show_steps=True),
    test_fgh('w+1', 3, FGH('w', FGH(2, f223), 2)),
    test_f_s('w+2'                , 3, 'w+2'  , test_only=True),
    test_f_s('w+w'                , 4, 'w+4'  , test_only=True),
    test_f_s('w*2'                , 3, 'w+3'  , show_step=True),
    test_fgh('w*2+1'              , 3),
    test_f_s('w*w'                , 4, 'w*3+4'  , test_only=True),
    test_f_s('v(0,1)'             , 3, '3'      , test_only=True),  # R2 v(0,g) = w^g
    test_f_s('v(0,2)'             , 3, 'w*2+3'  , show_step=True),  # R2 v(0,g) = w^g
    test_f_s('v(0,3)'             , 2, 'w^2+w+2', test_only=True),  # R2 v(0,g) = w^g
    test_f_s('w^2'                , 3, 'w*2+3'  , test_only=True),
    test_f_s('w^2'                , 4, 'w*3+4'),
    test_f_s('w*(w+1)'            , 3, 'w*w+3', test_only=True),
    test_f_s('w^2+w'              , 3, 'w^2+3'),
    test_fgh('w^2+w+1'            , 3),
    test_f_s('w^3'                , 4, 'w^2*3+w*3+4'),
    test_f_s('w^w'                , 3, 'w^2*2+w*2+3'),
    test_f_s('v(0,w)'             , 3, 'w^2*2+w*2+3', test_only=True),  # R2 v(0,g) = w^g
    FGH('w^w'  , 3).expand_once_display(FGH('w^2*2+w*2+3', 3), test_only=True),
    test_fgh('w^w'                , 3, test_only=True),
    test_f_s('w^(w+1)'            , 3, 'w^w*2 + w^2*2 + w*2 + 3', test_only=True),
    test_f_s('v(0,w+1)'           , 3, 'w^w*2 + w^2*2 + w*2 + 3', show_step=True),
    test_fgh('w^w+1'              , 3),
    test_f_s('v(0,w*2)'           , 3,
             '(((w^(w+2))*2)+(((w^(w+1))*2)+(((w^w)*2)+(((w^2)*2)+((w*2)+3)))))',
                                       print_str=True),
    test_f_s('w^((w^2)*2+w*2+2)'  , 3, test_only=True),

    (OutType.PLAIN, r'<h2> $ \varepsilon_0 $ </h2>'+'\n'),
    test_f_s('v(0,0)'             , 0, 0, test_only=True),  # R4
    test_f_s('v(1,0)'             , 0, 0),                  # R4
    test_f_s('v(2,0)'             , 0, 0, test_only=True),  # R4
    test_f_s('v(w,0)'             , 0, 0, test_only=True),  # R4
    test_f_s('v(1,0)'             , 1, 1),                  # R5
    # !! todo 1: v(0,) to w has repeating display; and missing index [2]
    test_f_s('v(1,0)'             , 2, 2, show_step=True),  # R5
    test_f_s('v(1,0)'             , 3,    test_only=True),  # R5
    # todo 2: more v tests
    test_f_s('e'                  , 3, 'w^2*2+w*2+3', show_step=True),
    # todo 2: w^w^w return if calc to the end. and assert, so limit isn't too small
    test_fgh('w^(w^w)'            , 2, show_step=True),
    # todo: smarter length ctrl based on terms
    test_fgh('w^(w^w)'            , 3, limit=4, show_step=True),
    test_f_s('e*w'                , 3),
    # test_f_s('e^e'              , 3),

    (OutType.PLAIN, r'<h2> $ \varphi(\alpha,\gamma) $ </h2>'+'\n'),
    test_f_s('v(1,1)'      , 0, 'e+1'    ),    # R6
    # todo: give a expected mid value for stop
    test_f_s('v(1,1)'      , 1, limit=4, show_step=True),  # R7 R6
    # test_f_s('v(1,1)'      , 1, 'w^(e+1)'),  # R7


  ]

  latex_to_html([s for s in tests if s is not None], './local_test.html')
  utils.print_total_time(Ord.rotate)
  print(f'  rotated {get_rotate_counter()} terms\n')

if __name__ == '__main__':
  res, _ = utils.timed(test_main)
  print(f"Tests done in {res:.2f} sec")
