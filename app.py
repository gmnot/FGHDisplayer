# app.py
from flask import Flask, request, render_template
from ordinal import Ord, FGH

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
  result = ''
  if request.method == 'POST':
    ord_str = request.form['large_text']
    n_str = request.form['small_text']
    action = request.form['action']

    if action == 'fund_seq':
        result = Ord.from_str(ord_str).fundamental_sequence_display(int(n_str))
    elif action == 'fgh':
        result = FGH(Ord.from_str(ord_str), int(n_str)).expand_display()

  return render_template('index.html', result=result)

if __name__ == '__main__':
  app.run(debug=True)
