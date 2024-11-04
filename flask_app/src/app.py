from flask import Flask, render_template
app = Flask(__name__)

@app.route('/')
def hello():
    return render_template('main.html')

@app.route('/generate')
def generateLogoImage():
    return render_template('generation_page.html')
    
@app.route('/myPage')
def myPage():
    return render_template('myPage.html')
if __name__ == '__main__':
    app.run(debug=True)

