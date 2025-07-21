"""
WICHTIG: DIESE DATEI NIEMALS ÄNDERN!
"""
import socket
import json
import os


def _connect(msg: dict):
    # save groupID
    msg["groupID"] = int(os.environ['JUPYTERHUB_USER'].split('_')[-1])

    # Create a socket object
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # Get the server hostname and port
    host = socket.gethostname()
    port = 12345

    # Connect to the server
    client_socket.connect((host, port))
    return client_socket


def _send_message(msg: dict):
    # connect to server
    client_socket = _connect(msg)

    # Serialize the dictionary to JSON
    message = json.dumps(msg)
    # Send the JSON-encoded message to the server
    client_socket.send(message.encode('utf-8'))
    # Receive response from the server, if JSON arrived
    response = client_socket.recv(1024).decode('utf-8')
    print("Response from server:", response)

    # If it is a Testlauf and no unexpected error occured,
    # receive results or error message
    if msg["Testlauf"] and response == "JSON arrived":
        print("waiting for response...")
        response = client_socket.recv(1024).decode('utf-8')
        print("Response from server:\n" + str(response))

    # Close the connection
    client_socket.close()


def test_submission(msg: dict):
    msg["final"] = False
    msg["Testlauf"] = True
    _send_message(msg)


def make_submission(msg: dict):
    msg["final"] = False
    msg["Testlauf"] = False
    _send_message(msg)


def final_submission(msg: dict):
    msg["final"] = True
    msg["Testlauf"] = False
    _send_message(msg)


def get_last_result():
    msg = {"run": 0}
    client_socket = _connect(msg)

    message = json.dumps(msg)
    client_socket.send(message.encode('utf-8'))
    response = client_socket.recv(1024).decode('utf-8')
    print("Response from server:", response)


def check_for_queue_and_errors():
    msg = {"check": True}
    client_socket = _connect(msg)

    message = json.dumps(msg)
    client_socket.send(message.encode('utf-8'))
    response = client_socket.recv(1024).decode('utf-8')
    print("Response from server:", response)
