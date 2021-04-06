import cv2
import numpy as np
from tensorflow.keras.models import load_model
import tensorflow as tf

def valid(x, y, n):
    for i in range(9):
        if board[x][i] == n:
            return False
    for i in range(9):
        if board[i][y] == n:
            return False
    x0 = (x//3)*3
    y0 = (y//3)*3
    for i in range(3):
        for j in range(3):
            if board[x0+i][y0+j] == n:
                return False
    return True

def solve():
    for i in range(9):
        for j in range(9):
            if board[i][j]==0:
                for n in range(1,10):
                    if valid(i,j,n):
                        board[i][j]=n
                        solve()
                        board[i][j]=0
                return
    print(board)

model = load_model('Model/myModel.h5')
full_img = cv2.imread("img/Sudoku.png")
full_img = cv2.resize(full_img, (450,450), interpolation= cv2.INTER_AREA)
height, width, channels = full_img.shape

#img to np board
board = np.array([])
a = 5
for i in range(9):
    row = np.array([])
    for j in range(9):
        x1, y1 = int(width * j / 9 + a), int(height * i / 9 + a)
        x2, y2 = int(width * (j + 1) / 9 - a), int(height * (i + 1) / 9 - a)
        img = full_img[y1:(y2 + 1), x1:(x2 + 1)]

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (28, 28), interpolation=cv2.INTER_AREA)
        newimg = tf.keras.utils.normalize(resized, axis=1)
        newimg = np.array(newimg).reshape(-1, 28, 28, 1)

        predictas = model.predict(newimg)
        row = np.append(row, np.argmax(predictas))
    board = np.append(board, row).astype('int32')
    board = board.reshape(9, -1)

solve()