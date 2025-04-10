from flask import Flask, render_template, request

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    if request.method == 'POST':
        large_text = request.form['large_text']
        small_text = request.form['small_text']
        action = request.form['action']

        if action == 'merge':
            # 合并字符串
            result = large_text + small_text
        elif action == 'reverse_merge':
            # 反向合并字符串
            result = (small_text + large_text)[::-1]  # 合并并反转

    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
