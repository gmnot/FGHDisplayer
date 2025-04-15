# app.py
from flask import Flask, request, render_template
from html_utils import latex_to_block, contact_request
from ordinal import Ord, FGH, FdmtSeq, KnownError, kraise, ord_set_debug_mode
from test import test_f_s, test_fgh

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
  result = ''
  if request.method == 'POST':
    try:
      ord_str = request.form['large_text']
      n_str = request.form['small_text']
      action = request.form['action']

      kraise(len(n_str) == 0, "Can't read index from empty string")
      kraise(not n_str.isdigit(), f"index {n_str} is not a number")
      n = int(n_str)

      if action == 'fund_seq':
        result = latex_to_block(test_f_s(ord_str, n, show_step=True))
      elif action == 'fgh':
        result = latex_to_block(test_fgh(ord_str, n, show_step=True))

    except KnownError as e:
      result = f'{e}'
    except Exception as e:
      if app.debug:
        result = f'Unknown error. Debugging: {e}'
      else:
        result = f"Unknown error, {contact_request}."

  return render_template('index.html', result=result)

if __name__ == '__main__':
  debug = True
  ord_set_debug_mode(debug)
  app.run(debug=debug)
