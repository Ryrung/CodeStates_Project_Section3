from flask_app import create_app

app = create_app()
app.run(debug=True)
app.run(host='127.0.0.1', debug=True)