from flask import Flask, request
import datetime

app = Flask(__name__)

@app.route("/health-check")
def hello():
    client_ip = request.environ.get('HTTP_X_REAL_IP', request.remote_addr)
    return f'Hello My Flask World! <br><br> Now: {datetime.datetime.now()} <br><br> Your IP: {client_ip}'

app.run(host="0.0.0.0", port="5000", debug=True)