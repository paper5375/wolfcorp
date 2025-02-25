from flask import Flask, request, Response, render_template
import time

app = Flask(__name__)

# Store the latest people count
latest_count = 0

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/update_count', methods=['POST'])
def update_count():
    global latest_count
    data = request.get_json()
    latest_count = data.get('people_count', 0)
    return {"status": "success"}, 200

@app.route('/stream')
def stream():
    def event_stream():
        while True:
            yield f"data: {latest_count}\n\n"
            time.sleep(1)  # Update every second
    return Response(event_stream(), mimetype="text/event-stream")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)