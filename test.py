from ordinal import Node, Ordinal, FGH

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
  expr_tree = Ordinal(
    Node('+',
         Node('*', Node('w'), Node('2')),
         Node('3'),
         )
  )
  print(expr_tree)  # Output: Infix Expression: (3 + (w * 2))

  latex_to_html([
    expr_tree.to_latex(),
    # Ordinal(Node('1')).fundamental_sequence_display(3, Node('1')),
    Ordinal(Node('w')).fundamental_sequence_display(3, Node('3')),
    # Ordinal.from_str('w+2').fundamental_sequence_display(3, Node.from_str('w+2')),
    Ordinal.from_str('w^2+w').fundamental_sequence_display(3, Node.from_str('w^2+3')),
    Ordinal.from_str('w*1').fundamental_sequence_display(3, Node.from_str('3')),
    Ordinal.from_str('w*2').fundamental_sequence_display(3, Node.from_str('w+3')),
    Ordinal.from_str('w*w').fundamental_sequence_display(4, Node.from_str('w*3+4')),
    Ordinal.from_str('w^1').fundamental_sequence_display(4, Node.from_str('4')),
    Ordinal.from_str('w^2').fundamental_sequence_display(3, Node.from_str('w*2+3')),
    Ordinal.from_str('w^w').fundamental_sequence_display(3, Node.from_str('w^2*2+(w*2+3)')),
    FGH(Ordinal.from_str('w^w'), 3).to_latex(),
    FGH(Ordinal.from_str('w^w'), FGH(Ordinal.from_str('w^w'), 3)).to_latex(),
    FGH(Ordinal.from_str('w^w'), 3).expand_once_display(FGH(Ordinal.from_str('w^2*2+(w*2+3)'), 3)),
    # FGH(Ordinal.from_str('w^(w+1)'), 3).expand_once().to_latex(),
  ], './test.html')
