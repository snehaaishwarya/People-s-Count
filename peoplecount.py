from flask import Flask, render_template, Response, request
import cv2

app = Flask(__name__)

# Load the pre-trained Haar cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Variables for tracking
people_in = 0
people_out = 0
tracked_faces = {}
face_id = 0

def generate_frames():
    global people_in, people_out, tracked_faces, face_id
    camera = cv2.VideoCapture(0)
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            height, width, _ = frame.shape
            mid_line = width // 2

            # Convert the frame to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Detect faces
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)

            new_tracked_faces = {}
            for (x, y, w, h) in faces:
                face_center_x = x + w // 2
                face_center_y = y + h // 2

                # Check if this face was tracked previously
                matched_id = None
                for id, (prev_x, prev_y) in tracked_faces.items():
                    if abs(face_center_x - prev_x) < w and abs(face_center_y - prev_y) < h:
                        matched_id = id
                        break

                if matched_id is None:
                    matched_id = face_id
                    face_id += 1

                new_tracked_faces[matched_id] = (face_center_x, face_center_y)

                # Draw rectangle around the faces
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

                # Check crossing of the middle line
                if matched_id in tracked_faces:
                    prev_x, prev_y = tracked_faces[matched_id]
                    if prev_x <= mid_line and face_center_x > mid_line:
                        people_in += 1
                    elif prev_x > mid_line and face_center_x <= mid_line:
                        people_out += 1

            tracked_faces = new_tracked_faces

            # Draw the middle line
            cv2.line(frame, (mid_line, 0), (mid_line, height), (255, 0, 0), 2)

            # Display counts
            cv2.putText(frame, f'Out: {people_in}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, f'In: {people_out}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            # Encode the frame in JPEG format
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            # Yield the frame in byte format
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start')
def start():
    return '', 204

if __name__ == '__main__':
    app.run(debug=True)
