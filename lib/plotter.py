import numpy as np
import cv2

def plot_original_vs_generated(original, generated):
    num_images = 15
    sample = np.random.randint(0, len(original), num_images)

    def stack(data):
        images = data[sample]

        return np.vstack(
                [
                    np.hstack(images[5]),
                    np.hstack(images[5:10]),
                    np.hstack(images[10:15]),

                ]
            )

    def add_text(image, text, position):
        pt1 = position
        pt2 = (pt1[0] + 10 + (len(text) * 22),
                pt1[1] - 45)

        cv2.rectangle(image,
                pt1,
                pt2,
                (255, 255, 255),
                -1)

        cv2.putText(
                image,
                text,
                position,
                fontFace = cv2.FONT_HERSHEY_SIMPLEX,
                fontScale = 1.3,
                color = (0, 0, 0),
                thickness = 4
                )

    original = stack(original)
    generated = stack(generated)

    mosaic = np.vstack([original, generated])
    mosaic = cv2.resize(mosaic, (860, 860),
            interpolation = cv2.INTER_AREA)

    mosaic = cv2.cvtColor(mosaic, cv2.COLOR2BGR)

    add_text(mosaic, "Original", (50 , 100))
    add_text(mosaic, "Generated", (50 , 520))

    cv2.imshow("Mosaic", mosaic)
    cv2.waitKey(0)

