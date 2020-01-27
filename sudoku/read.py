""" This script takes a sudoku image as input, solves it and puts answer on image itself.
Usage : python sudoku.py <image_file> """
import cv2
import numpy as np
from solve import solve_sudoku

# Load OCR data for training
samples = np.float32(np.loadtxt('sudoku/feature_vector_pixels.data'))
responses = np.float32(np.loadtxt('sudoku/samples_pixels.data'))

model = cv2.ml.KNearest_create()
model.train(samples, cv2.ml.ROW_SAMPLE, responses)


#  Function to put vertices in clockwise order
def rectify(h):
    """ this function put vertices of square we got, in clockwise order """
    h = h.reshape((-1, 2))
    hnew = np.zeros((4, 2), dtype=np.float32)

    add = h.sum(1)
    hnew[0] = h[np.argmin(add)]
    hnew[2] = h[np.argmax(add)]

    diff = np.diff(h, axis=1)
    hnew[1] = h[np.argmin(diff)]
    hnew[3] = h[np.argmax(diff)]

    return hnew


def solve_from_image(image_path):
    # Now starts main program
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    thresh = cv2.adaptiveThreshold(gray, 255, 1, 1, 5, 2)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    image_area = gray.size  # this is area of the image
    for i in contours:
        if cv2.contourArea(i) > image_area / 2:  # if area of box > half of image area, it is possibly the biggest blob
            peri = cv2.arcLength(i, True)
            approx = cv2.approxPolyDP(i, 0.02 * peri, True)
            # cv2.drawContours(img,[approx],0,(0,255,0),2,cv2.CV_AA)
            break
    # Now we got sudoku boundaries, Transform it to perfect square

    h = np.array([[0, 0], [449, 0], [449, 449], [0, 449]],
                 np.float32)  # this is corners of new square image taken in CW order

    approx = rectify(approx)  # we put the corners of biggest square in CW order to match with h

    retval = cv2.getPerspectiveTransform(approx, h)  # apply perspective transformation
    warp = cv2.warpPerspective(img, retval, (450, 450))  # Now we get perfect square with size 450x450
    warpg = cv2.cvtColor(warp, cv2.COLOR_BGR2GRAY)  # kept a gray-scale copy of warp for further use
    sudo = np.zeros((9, 9), np.uint8)  # a 9x9 matrix to store our sudoku puzzle
    smooth = cv2.GaussianBlur(warpg, (3, 3), 3)
    thresh = cv2.adaptiveThreshold(smooth, 255, 0, 1, 5, 2)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    erode = cv2.erode(thresh, kernel, iterations=1)
    dilate = cv2.dilate(erode, kernel, iterations=1)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 100 < area < 800:

            (bx, by, bw, bh) = cv2.boundingRect(cnt)
            if (100 < bw * bh < 1200) and (10 < bw < 40) and (25 < bh < 45):
                roi = dilate[by:by + bh, bx:bx + bw]
                small_roi = cv2.resize(roi, (10, 10))
                feature = small_roi.reshape((1, 100)).astype(np.float32)
                ret, results, neigh, dist = model.findNearest(feature, k=1)
                integer = int(results.ravel()[0])

                gridy, gridx = int((bx + bw / 2) / 50), int(
                    (by + bh / 2) / 50)  # gridx and gridy are indices of row and column in sudo
                sudo.itemset((gridx, gridy), integer)

    sudoku_area = ''.join(str(n) for n in sudo.flatten())
    # How many rectangle in one column?
    n = 9

    """ It will be return this:
        [[0, 0, 0, 0, 5, 0, 0, 0, 0],
         [7, 0, 0, 0, 0, 0, 0, 0, 4],
         [0, 0, 0, 8, 0, 2, 0, 0, 0],
         [0, 2, 0, 0, 8, 0, 0, 3, 0],
         [0, 0, 0, 6, 0, 3, 0, 0, 0],
         [3, 4, 0, 0, 0, 0, 0, 9, 8],
         [8, 0, 9, 7, 0, 6, 2, 0, 3],
         [0, 0, 1, 0, 4, 0, 9, 0, 0],
         [4, 7, 0, 0, 0, 0, 0, 8, 5]] """
    prepare_for_solve = [list(map(int, sudoku_area[i:i + n])) for i in range(0, len(sudoku_area), n)]
    return solve_sudoku(prepare_for_solve)


if __name__ == '__main__':
    import sys
    solve_from_image(sys.argv[1])
