from enum import Enum
from html_utils import OutType
import ordinal
from ordinal import calc_display, get_rotate_counter, FdmtSeq, FGH, \
                    ord_set_debug_mode, Ord, WIPError
from veblen import Veblen, parse_v_list
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

def test_f_s(ord1 : str | int, n : int, expected=None, **kwargs):

  return calc_display(
    FdmtSeq(Ord.clean_str(ord1) if isinstance(ord1, str) else ord1,
            n, latex_force_veblen=True),
    expected, **kwargs
  )

def test_fgh(ord1 : str | int, n : int, expected=None, **kwargs):

  return calc_display(
    FGH(Ord.clean_str(ord1) if isinstance(ord1, str) else ord1,
        n, latex_force_veblen=True),
    expected, **kwargs
  )

def test_basics():
  assert parse_v_list("v(1,0,0)") == parse_v_list("v(1@2)")
  assert parse_v_list("v((w^3)[3],0)") == parse_v_list("v((w^3)[3]@1)")

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

  test_basics()
  test_associative()
  # test_exceptions()

  tests = [
    expr_tree.to_latex(),
    (OutType.PLAIN, r'<h2> $ f_c(n) $ </h2>'+'\n'),
    test_fgh(0, 3  , 4),
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
    test_f_s('v（0,0)'            , 0,  0,        test_only=True),  # R4
    test_f_s('v(0，1)'            , 3, '3'      , test_only=True),  # R2 v(0,g) = w^g
    test_f_s('v(0,2）'            , 3, 'w*2+3'  , show_step=True),  # R2 v(0,g) = w^g
    test_f_s('v(0, 3)\n'          , 2, 'w^2+w+2', test_only=True),  # R2 v(0,g) = w^g
    test_f_s('w^2\t'              , 3, 'w*2+3'  , test_only=True),
    test_f_s('w^2'                , 4, 'w*3+4'),
    test_f_s('w*(w+1)'            , 3, 'w*w+3', test_only=True),
    test_f_s('w^2+w'              , 3, 'w^2+3'),
    test_fgh('w^2+w+1'            , 3),
    test_f_s('w^3'                , 4, 'w^2*3+w*3+4'),
    test_f_s('w^w'                , 3, 'w^2*2+w*2+3'),
    test_f_s('v(0,w)'             , 3, 'w^2*2+w*2+3', test_only=True),  # R2 v(0,g) = w^g
    test_fgh('w^w'                , 3, until=FGH('w^2*2+w*2+3', 3)),
    test_fgh('w^w'                , 3, test_only=True),
    test_f_s('w^(w+1)'            , 3, 'w^w*2 + w^2*2 + w*2 + 3', test_only=True),
    test_f_s('v(0,w+1)'           , 3, 'w^w*2 + w^2*2 + w*2 + 3', show_step=True),
    test_fgh('w^w+1'              , 3),
    test_f_s('v(0,w*2)'           , 3,
             '(((w^(w+2))*2)+(((w^(w+1))*2)+(((w^w)*2)+(((w^2)*2)+((w*2)+3)))))'),
    test_f_s('w^((w^2)*2+w*2+2)'  , 3, test_only=True),

    (OutType.PLAIN, r'<h2> $ \varepsilon_0 $ </h2>'+'\n'),
    test_f_s('v(1,0)'             , 0, 0),                  # R4
    test_f_s('v(2,0)'             , 0, 0, test_only=True),  # R4
    test_f_s('v(w,0)'             , 0, 0, test_only=True),  # R4
    test_f_s('v(1,0)'             , 1, 1),                  # R5 v(a+1,0)
    test_f_s('v(1,0)'             , 2, 2, show_step=True),  # R5 v(a+1,0)
    test_f_s('v(1,0)'             , 3,    test_only=True),  # R5 v(a+1,0)
    test_f_s('e'                  , 3, 'w^2*2+w*2+3', show_step=True),
    test_fgh('w^(w^w)'            , 2, show_step=True),
    test_fgh('w^(w^w)'            , 3, limit=4, show_step=True),
    test_f_s('e*w'                , 3),
    # test_f_s('e^e'              , 3),

    (OutType.PLAIN, r'<h2> $ \varphi(\alpha,\gamma) $ </h2>'+'\n'),
    test_f_s('v(1,1)'      , 0, 'e+1'),                  # R6
    test_f_s('v(1,1)'      , 1, until=FdmtSeq('w^(e+1)', 1), limit=20,
                                show_step=True),         # R7 R6 v(a+1,g+1)[n+1 to 0]
    test_f_s('v(1,1)'      , 2, until=Veblen(0, FdmtSeq('w^(v(1, 0)+1)', 2)),
                                show_step=True),         # R7 R6
    # FGH for Ord close to cal_limit. Shouldn't be too long
    test_fgh('w^((w^2)*2)' , 3),
    test_f_s('v(1,w)'      , 2, until=Veblen(0, FdmtSeq('w^(v(1, 1)+1)', 2)),
                                test_only=True),         # R3 R7 R6, g is LO
    test_f_s('v(1,v(0,1))' , 2, until=Veblen(0, FdmtSeq('w^(v(1, 1)+1)', 2)),
                                show_step=True),
    test_f_s('v(2,0)'      , 3, until="v(1, v(1, v(0, v(1,0)[2])))",
                                show_step=True),         # R5 v(a+1,0)
    test_f_s('v(2,w)'      , 2, until="v(1, v(0, (w^(v(1, v(2,1))+1))[2]))",
                                show_step=True),         # R3 R7 R6, g is LO
    # todo 3: better display for long power / sub
    test_fgh('v(4,1)'      , 3, show_step=True),         # R3 R7 R6, g is LO
    test_f_s('v(w,0)'      , 2, until="v(1,2)[2]",
                                show_step=True),         # R8 v(a, 0)
    test_f_s('v(w,0)'      , 3, until="v(2, v(2, v(1, v(1, v(0, w[3])))))",
                                show_step=True),         # R8 v(a, 0)
    test_f_s('v(w,1)'      , 2, until="v(1, v(0, ((w^v(1, v(2, v(w, 0))))*w[2])))",
                                show_step=True),         # R9 v(a, g+1)
    test_f_s('v(w,w)'      , 2,
             until="v(1, v(0, ((w^v(1, v(2, v(w, 1))))+(w^v(1, v(2, v(w, 1)[2]))))))",
                                show_step=True),         # R9 v(a, g+1)
    test_f_s('v(w,2)'      , 3,
             until="v(2, v(2, v(1, v(1, v(1, v(2, (v(3, v(w, 1))+1))[0])))))",
                                show_step=True),         # R9 v(a, g+1)

    (OutType.PLAIN, r'<h2> $ \Gamma_0, \ \varphi(\alpha,\beta,\dots,\gamma) $ </h2>'+'\n'),
    test_f_s('v(v(v(w,0),0),0)' , 3,
             until="v(v(v(2, v(2, v(1, v(1, v(0, w[3]))))), 0),0)",
                                show_step=True),
    test_f_s('v(3)'        , 2, until="(w^3)[2]"),       # R1 v(g+1)
    test_f_s('v(w)'        , 3, until="(w^3)[3]",
                                test_only=True),         # R2 v(g)
    test_f_s('v(v(2)+v(1))', 3, until="((w^(v(2)+2))*w)[3]",
                                test_only=True),
    test_f_s('v(1,0,0)'    , 0, expected=0,
                                test_only=True),         # R3-1 v(S,a+1,Z,0)[0] = 0
    test_f_s('v(1@2)'      , 0, expected=0,
                                test_only=True),         # R3
    test_f_s('v(4,0,0)'    , 0, expected=0),             # R3-1 v(S,a+1,Z,0)[0] = 0
    test_f_s('v(5@2)'      , 0, expected=0,
                                test_only=True),         # R3
    # R3-2 v(S,a+1,Z,0)[n+1] = v(S,a,v(S,a+1,Z,0)[n],Z)
    # R6 v(S,a,Z,0)[n] = v(S,a[n],Z,0)
    test_f_s('v(1,0,0)'    , 3, until="v((w^3)[3],0)",
                                test_only=True),
    test_f_s('v(1@2)'      , 3, until="v((w^3)[3]@1)",
                                test_only=True),
    test_f_s('v(1,0,0)'    , 3, limit=65,
                                show_step=True),
    # R4 v(S,a+1,Z,g+1): b -> v(S,a,b,Z)
    test_f_s('v(1,0,1)'    , 3, until="v(v(v(v(1,0,0),v((v(1,0,0)+1),0)[2]),0),0)",
                                show_step=True),
    test_f_s('v(1@2,1@0)'  , 3, until="v(v(v(v(1,0,0),v((v(1,0,0)+1),0)[2]),0),0)",
                                test_only=True),  # @R4
    # R5 R4
    test_f_s('v(1,0,w)'    , 3, until="v(v(1,0,3)[2],0)",
                                show_step=True),
    test_f_s('v(1@2,w@0)'  , 3, until="v(v(1,0,3)[2],0)",
                                test_only=True),  # @R2: g is LO; @R4
    test_f_s('v(1,0,e)'    , 3, until="v(1,0,(((w^2)*2)+((w*2)+w)[3]))",
                                test_only=True),
    test_f_s('v(1@2,e@0)'  , 3, until="v(1,0,(((w^2)*2)+((w*2)+w)[3]))",
                                test_only=True),  # @R2: g is LO; @R4
    test_f_s('v(1,0,v(1,0,0))'  , 3, until="v(1,0,v(v(0,v(1,0)[2]),0))",
                                     show_step=True),
    test_f_s('v(1@2,v(1@2)@0)'  , 3, until="v(1,0,v(v(0,v(1,0)[2]),0))",
                                     test_only=True),  # @R2: g is LO
    test_f_s('v(1,1,0)'         , 3, until="v(1,0,v(1,0,v(v(1,0,0)[2],0)))",
                                     show_step=True),
    test_f_s('v(1@2,1@1)'       , 3, until="v(1,0,v(1,0,v(v(1,0,0)[2],0)))",
                                     test_only=True),
    test_f_s('v(1,w,0)'         , 3, until="v(1,2,v(1,2,v(1,1,v(1,1,v(1,2,0)[1]))))",
                                     show_step=True),
    test_f_s('v(1@2,w@1)'       , 3, until="v(1,2,v(1,2,v(1,1,v(1,1,v(1,2,0)[1]))))",
                                     test_only=True),
    test_f_s('v(1,v(1,0,1),0)'  , 3, until="v(1,v(v(v(v(1,0,0),v((v(1,0,0)+1),0)[2]),0),0),0)"),
    test_f_s('v(1@2,v(1,0,1)@1)', 3, until="v(1,v(v(v(v(1,0,0),v((v(1,0,0)+1),0)[2]),0),0),0)",
                                     test_only=True),
    test_f_s('v(1,v(1,1,0),0)'  , 3, until="v(1,v(1,0,v(1,0,v(v(1,0,0)[2],0))),0)"),
    test_f_s('v(1@2,v(1,1,0)@1)', 3, until="v(1,v(1,0,v(1,0,v(v(1,0,0)[2],0))),0)",
                                     test_only=True),
    # MV R3 R6
    # @R3 @R7
    test_f_s('v(2,0,0)'         , 3, until="v(1,v(1,v(v(1,0,0)[2],0),0),0)",
                                     test_only=True),
    test_f_s('v(2@2)'           , 3, until="v(1,v(1,v(v(1,0,0)[2],0),0),0)",
                                     test_only=True),
    test_f_s('v(w,0,0)'         , 3, until="v(2,v(2,v(1,v(1,v(v(v(1,0,0)[1],0),0),0),0),0),0)",),
    test_f_s('v(w@2)'           , 3, until="v(2,v(2,v(1,v(1,v(v(v(1,0,0)[1],0),0),0),0),0),0)",
                                     test_only=True),
    test_f_s('v(w+1,0,0)'       , 3, until="v(w,v(w,v(2,v(2,v(1,v(1,v(1,0,0)"
                                           "[3],0),0),0),0),0),0)",),
    test_f_s('v(w+1@2)'         , 3, until="v(w,v(w,v(2,v(2,v(1,v(1,v(1,0,0)"
                                           "[3],0),0),0),0),0),0)",
                                     test_only=True),
    # MV R7 v(S,a,Z,g+1)[n] = v(S,a[n],Z,(S,a,Z,g)+1)
    # @Rx
    test_f_s('v(w,0,1)'         , 3, until="v(3,(v(w,0,0)+1),0)[3]",
                                     show_step=True),
    test_f_s('v(w@2,1@0)'       , 3, until="v(3,(v(w,0,0)+1),0)[3]",
                                     test_only=True),
    test_f_s('v(1,w,0,1)'       , 3, until="v(1,3,v(1,w,0,0),v(1,3,v(1,w,0,0),"
                                           "v(1,3,v(1,3,0,0)[3],0)))",
                                     test_only=True),
    test_f_s('v(1@3,w@2,1@0)'   , 3, until="v(1,3,v(1,w,0,0),v(1,3,v(1,w,0,0),"
                                           "v(1,3,v(1,3,0,0)[3],0)))",
                                     test_only=True),
    test_f_s('v(w,w,0,1)'       , 3, until="v(w,3,v(w,w,0,0),v(w,3,v(w,w,0,0),"
                                           "v(w,3,v(w,w,0,0),v(w,3,(v(w,w,0,0)+1),0)[0])))",
                                     test_only=True),
    test_f_s('v(w@3,w@2,1@0)'   , 3, until="v(w,3,v(w,w,0,0),v(w,3,v(w,w,0,0),"
                                           "v(w,3,v(w,w,0,0),v(w,3,(v(w,w,0,0)+1),0)[0])))",
                                     test_only=True),
    # R8 v(S,a,Z,g[n])
    test_f_s('v(e,w,0,w)'       , 3, until="v(v(1,0),3,v(v(1,0),w,0,2),v(v(1,0),3,"
                                           "(v(v(1,0),w,0,2)+1),0)[2])",
                                     show_step=True),

    test_f_s('v(1,0,0,0,0)'     , 3, until="v(v(v(v(1,0,0)[2],0),0,0),0,0,0)",
                                     show_step=True),

    (OutType.PLAIN, r'<h2> $ \varphi(\alpha \mathbin{\char64} \beta) $ </h2>'+'\n'),
    # @R1: v(g) = w^g
    test_f_s('v(w@0)'           , 3, until='(w^w)[3]'),
    # @R5
    test_f_s('v(1@w)'           , 3, until='v((1@3))[3]'),
    test_f_s('v(2@w)'           , 3, until='v((1@w),(v((1@w),(v((1@w))[3]@2))@2))',
                                     show_step=True),
    test_f_s('v(1@e)'           , 3, until='v((1@(((w^2)*2)+(w*w)[3])))',
                                     test_only=True),
    test_f_s('v(1@w+1)'         , 3, until='v((v((1@3))[3]@w))',
                                     test_only=True),
    # @R6
    test_f_s('v(1@w,1@0)'       , 3, until='v((v((1@w))@3),(v(((v((1@w))+1)@3))[2]@2))',
                                     show_step=True),  # also @R3.1
    test_f_s('v(2@w,w@0)'       , 3, until='v((1@w),((v((2@w),(2@0))+1)@3))[3]',
                                     show_step=True),
    # @R7
    test_f_s('v(w@w)'           , 3, until='v((2@w),(1@3))[3]',
                                     test_only=True),
    test_f_s('v(e@e)'           , 3, until='v((((((w^2)*2)+(w*2))+2)@v(1,0)),(1@v(1,0)[3]))',),
    # @R8
    test_f_s('v(w@w,1@0)'       , 1, until='1',
                                     test_only=True),
    test_f_s('v(w@w,1@0)'       , 3, until='v((w[3]@w),(v((w@w))@3),(v((w[3]@w),'
                                           '((v((w@w))+1)@3))[2]@2))',
                                     show_step=True),
    test_f_s('v(1@v(1@w))'      , 3, until='v((1@v((v((((((w^2)*2)+(w*2))+2)@1),'
                                           '(v((((((w^2)*2)+(w*2))+3)@1))[2]@0))@2))))',),


    # ! template
    # test_f_s('v(w@w,1@0) '      , 3, print_str=True, n_steps=50,
    #                                  show_step=True),
  ]

  latex_to_html([s for s in tests if s is not None], './local_test.html')
  utils.print_total_time(Ord.rotate)
  print(f'  rotated {get_rotate_counter()} terms\n')

if __name__ == '__main__':
  res, _ = utils.timed(test_main)
  print(f"Tests done in {res:.2f} sec")
