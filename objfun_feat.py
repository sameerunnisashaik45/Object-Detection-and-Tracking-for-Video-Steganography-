import numpy as np
from Global_Vars import Global_Vars
import cv2 as cv
from scipy.stats import kurtosis


def objfun(Soln):
    Image = Global_Vars.Image
    Fitn = np.zeros(Soln.shape[0])
    dimension = len(Soln.shape)
    if dimension == 2:
        for i in range(Soln.shape[0]):
            sol = Soln[i, :]
            inten = []
            kurto = []
            for n in range(len(Image)):
                mog = cv.createBackgroundSubtractorMOG2()
                fgmask = mog.apply(Image[n])

                kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
                fgmask = cv.erode(fgmask, kernel, iterations=1)
                fgmask = cv.dilate(fgmask, kernel, iterations=1)

                contours, hierarchy = cv.findContours(fgmask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

                for contour in contours:
                    # Ignore small contours
                    if cv.contourArea(contour) < 1000:
                        continue

                    # Draw bounding box around contour
                    x, y, w, h = cv.boundingRect(contour)
                    cv.rectangle(Image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    flat = Image.flatten()
                _, thresh = cv.threshold(Image, 127, 255, cv.THRESH_BINARY)

                # Find contours
                contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
                for contour in contours:
                    # Get bounding box for the region
                    x, y, w, h = cv.boundingRect(contour)
                    # Extract the region of interest (ROI)
                    roi = Image[sol:sol + h, x:x + w]
                    # Calculate Intensity (Mean Pixel Value)
                    intensity = np.mean(roi)
                    inten.append(intensity)

                    # Calculate Kurtosis
                    # Convert ROI to grayscale if it's not already
                    if len(roi.shape) == 3:
                        roi_gray = cv.cvtColor(roi[n], cv.COLOR_BGR2GRAY)
                    else:
                        roi_gray = roi
                    # Flatten the ROI for kurtosis calculation
                    roi_gray_flat = roi_gray.flatten()

                    # Calculate kurtosis
                    kurtos = kurtosis(roi_gray_flat)
                    kurto.append(kurtos)
            Fitn[i] = 1/(np.asarray(inten) + np.asarray(kurto))
        return Fitn
    else:
        sol = Soln
        inten = []
        kurto = []
        for n in range(len(Image)):
            mog = cv.createBackgroundSubtractorMOG2()
            fgmask = mog.apply(Image[n])

            kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
            fgmask = cv.erode(fgmask, kernel, iterations=1)
            fgmask = cv.dilate(fgmask, kernel, iterations=1)

            contours, hierarchy = cv.findContours(fgmask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                # Ignore small contours
                if cv.contourArea(contour) < 1000:
                    continue

                # Draw bounding box around contour
                x, y, w, h = cv.boundingRect(contour)
                cv.rectangle(Image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                flat = Image.flatten()
            _, thresh = cv.threshold(Image, 127, 255, cv.THRESH_BINARY)

            # Find contours
            contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                # Get bounding box for the region
                x, y, w, h = cv.boundingRect(contour)
                # Extract the region of interest (ROI)
                roi = Image[sol:sol + h, x:x + w]
                # Calculate Intensity (Mean Pixel Value)
                intensity = np.mean(roi)
                inten.append(intensity)

                # Calculate Kurtosis
                # Convert ROI to grayscale if it's not already
                if len(roi.shape) == 3:
                    roi_gray = cv.cvtColor(roi[n], cv.COLOR_BGR2GRAY)
                else:
                    roi_gray = roi
                # Flatten the ROI for kurtosis calculation
                roi_gray_flat = roi_gray.flatten()

                # Calculate kurtosis
                kurtos = kurtosis(roi_gray_flat)
                kurto.append(kurtos)
        Fitn = 1 / (np.asarray(inten) + np.asarray(kurto))
        return Fitn
