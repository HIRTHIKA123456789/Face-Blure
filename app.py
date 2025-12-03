import cv2
from flask import Flask, Response, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
video_capture = cv2.VideoCapture(0)
processing = False

def process_frames():
    global processing
    while processing:
        check, frame = video_capture.read()
        if not check:
            break

        gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5)

        for (x, y, w, h) in faces:
            # Blur faces
            frame[y:y+h, x:x+w] = cv2.medianBlur(frame[y:y+h, x:x+w], 35)

        # Convert frame to JPEG format for transmission
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/start_processing', methods=['POST'])
def start_processing():
    global processing
    if not processing:
        processing = True
        return jsonify({"message": "Processing started", "video_url": "/video_feed"})
    else:
        return jsonify({"message": "Processing already started"})

@app.route('/stop_processing', methods=['POST'])
def stop_processing():
    global processing
    if processing:
        processing = False
        video_capture.release()
        cv2.destroyAllWindows()
        return jsonify({"message": "Processing stopped"})
    else:
        return jsonify({"message": "Processing not started"})

@app.route('/video_feed')
def video_feed():
    return Response(process_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
