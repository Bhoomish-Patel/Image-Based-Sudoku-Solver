import cv2
import numpy as np
import tensorflow as tf
model = tf.keras.models.load_model('myModel.h5')    

def is_valid(board, row, col, num):
    for i in range(9):
        if board[row][i] == num:
            return False
    for i in range(9):
        if board[i][col] == num:
            return False
    start_row, start_col = 3 * (row // 3), 3 * (col // 3)
    for i in range(3):
        for j in range(3):
            if board[start_row + i][start_col + j] == num:
                return False

    return True
def solve_sudoku(board):
    empty = find_empty(board)
    if not empty:
        return True 
    row, col = empty
    for num in range(1, 10):
        if is_valid(board, row, col, num):
            board[row][col] = num
            if solve_sudoku(board):
                return True
            board[row][col] = 0 
    return False

def find_empty(board):
    for i in range(9):
        for j in range(9):
            if board[i][j] == 0:
                return (i, j) 

    return None

def print_board(board):
    for row in board:
        print(" ".join(str(num) for num in row))

def extract_grid(image, points):
    dst_points = np.array([[0, 0], [450, 0], [450, 450], [0, 450]], dtype="float32")
    M = cv2.getPerspectiveTransform(np.array(points, dtype="float32"), dst_points)
    warped = cv2.warpPerspective(image, M, (450, 450))
    warped=cv2.cvtColor(warped,cv2.COLOR_BGR2GRAY)
    return warped

def sort_points(points):
    points = sorted(points, key=lambda x: x[1])
    if points[0][0] > points[1][0]:
        points[0], points[1] = points[1], points[0]
    if points[2][0] < points[3][0]:
        points[2], points[3] = points[3], points[2]
    return points

def get_approx_contour(contour):
    epsilon = 0.1 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    return approx

def extract_cells(grid):
    Matrix = [[0 for x in range(9)] for y in range(9)]
    rows=np.vsplit(grid,9)
    row=0
    for r in rows:
        cols=np.hsplit(r,9)
        col=0
        for c in cols:
            c=np.asarray(c)
            c=c[4:c.shape[0]-4,4:c.shape[1]-4]
            c=cv2.resize(c,(28,28))
            c=c/255
            c=c.reshape(1,28,28,1)
            predictions=model.predict(c)
            digit=np.argmax(predictions)
            probabilityValue=np.amax(predictions)
            if probabilityValue>0.8:
                Matrix[row][col]=digit
            col=col+1
        row=row+1
    print(Matrix)
    return Matrix

def process_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    threshold = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV, 11, 2)
    return threshold

def get_mx_contour(threshold):
    contours,_=cv2.findContours(threshold,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cA=0
    for contour in contours:
        approx=get_approx_contour(contour)
        if(len(approx)==4):
            if(cv2.contourArea(contour)>cA):
                mxcontour=contour
                cA=cv2.contourArea(contour)
    return mxcontour
    
def main():
    image = cv2.imread('img2.png')
    image=cv2.resize(image,(450,450))
    threshold=process_image(image)
    mxcontour=get_mx_contour(threshold)
    approx = get_approx_contour(mxcontour)
    points=approx.reshape(4,2)
    points=sort_points(points)
    result=extract_grid(image,points)
    sudoku=extract_cells(result)
    if solve_sudoku(sudoku):
        print("Sudoku solved successfully!")
        print_board(sudoku)
    else:
        print("No solution exists for the given Sudoku.")

if __name__ == "__main__":
    main() 