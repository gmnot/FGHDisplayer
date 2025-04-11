# app.py
from flask import Flask, request, render_template
from ordinal import Ord, FGH, KnownError, kraise, ord_set_debug_mode
from html_utils import latex_to_block, contact_request

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
        result = latex_to_block(Ord.from_str(ord_str).fundamental_sequence_display(
                                n, show_steps=True))
      elif action == 'fgh':
        result = latex_to_block(FGH(Ord.from_str(ord_str), n).expand_display())

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
