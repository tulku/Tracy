import zmq
import numpy
import cv2

host="192.168.1.177"
port=1337


def connect_to_screen(host,port):    
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect(f"tcp://{host}:{port}")
    return socket


def open_image(img_path= 'tag16_05.png'):
    img = cv2.imread(img_path)
    cv2.imshow("QR",img)
    return img

def send_img(socket, img):
    socket.send(img.data.tobytes())
    socket.recv()
    
socket = connect_to_screen(host,port)
img = open_image()

while True:
    # numbers = numpy.random.bytes(12288)
    send_img(socket,img)
    #    print(f"reply: {reply}")
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

