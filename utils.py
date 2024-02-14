import glob as glob
import matplotlib.pyplot as plt
import cv2
import numpy as np
import torch
import random
try:
    from healpy.newvisufunc import projview, newprojplot
except:
    pass
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator


def hex_to_rgb(hex_color):
    """convert hex string to rgb tuple

    Args:
        hex_color (str): some hex code thing like #A1876B

    Returns:
        tuple(uint8, uint8, uint8): r g b color
    """    
    # Remove the '#' if present
    hex_color = hex_color.lstrip('#')
    
    # Convert the hex color to RGB
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)

    return b, g, r


def angle_loss(output, target):
    # output_angle = output * torch.pi / 180
    # target_angle = target * torch.pi / 180
    # loss = torch.mean((torch.cos(output_angle) - torch.cos(target_angle))**2 + \
    #                   (torch.sin(output_angle) - torch.sin(target_angle))**2)
    loss = torch.mean(torch.min(torch.abs(output - target), torch.abs(360 + output - target)))
    return loss

class LinearNDInterpolatorExt(object):
    def __init__(self, points,values):
        self.funcinterp = LinearNDInterpolator(points,values)
        self.funcnearest = NearestNDInterpolator(points,values)
    def __call__(self,*args):
        t = self.funcinterp(*args)
        if not np.isnan(t):
            return t.item(0)
        else:
            return self.funcnearest(*args)


class_names = ['star', 'BH']
# colors = np.random.uniform(0, 255, size=(len(class_names), 3))
colors = ['#D65DB1', '#FF9671']
colors = list(map(hex_to_rgb, colors))

# Function to convert bounding boxes in YOLO format to xmin, ymin, xmax, ymax.
def yolo2bbox(bboxes):
    """convert yolo coordinate to x and y of two points

    Args:
        bboxes (tuple): _description_

    Returns:
        _type_: _description_
    """    
    xmin, ymin = bboxes[0]-bboxes[2]/2, bboxes[1]-bboxes[3]/2
    xmax, ymax = bboxes[0]+bboxes[2]/2, bboxes[1]+bboxes[3]/2
    return xmin, ymin, xmax, ymax


def plot_box(image, bboxes, labels):
    """_summary_

    Args:
        image (np.ndarray): _description_
        bboxes (list): _description_
        labels (_type_): _description_

    Returns:
        numpy.ndarray: images with bounding boxes
    """    
    # Need the image height and width to denormalize
    # the bounding box coordinates
    h, w, _ = image.shape
    for box_num, box in enumerate(bboxes):
        x1, y1, x2, y2 = yolo2bbox(box)
        # denormalize the coordinates
        xmin = int(x1*w)
        ymin = int(y1*h)
        xmax = int(x2*w)
        ymax = int(y2*h)
        width = xmax - xmin
        height = ymax - ymin

        class_name = class_names[int(labels[box_num])]

        cv2.rectangle(image,(xmin, ymin), (xmax, ymax),
            color=colors[class_names.index(class_name)], thickness=2
        )

        font_scale = min(1,max(3,int(w/500)))
        font_thickness = min(2, max(10,int(w/50)))

        p1, p2 = (int(xmin), int(ymin)), (int(xmax), int(ymax))
        # Text width and height
        tw, th = cv2.getTextSize(class_name, 0, fontScale=font_scale, thickness=font_thickness)[0]
        p2 = p1[0] + tw, p1[1] + -th - 10
        cv2.rectangle(image, p1, p2, color=colors[class_names.index(class_name)], thickness=-1)
        cv2.putText(image, class_name, (xmin+1, ymin-10), cv2.FONT_HERSHEY_SIMPLEX,
            font_scale, (255, 255, 255), font_thickness
        )
    return image

def plot_circle(image, circles, labels):
    """plot circle on the image

    Args:
        image (np.ndarray): input image
        circles (list): coord of all circles
        labels (list): labels of all circles

    Returns:
        np.ndarray: the image after plot
    """
    # Need the image height and width to denormalize
    # the bounding box coordinates
    h, w, _ = image.shape
    assert h == w, 'Only support square imagaes !!!'
    for box_num, box in enumerate(circles):
        x, y, r, _ = box
        # denormalize the coordinates
        x = int(x*w)
        y = int(y*h)
        r = int(r*w)

        class_name = class_names[int(labels[box_num])]

        cv2.circle(
            image, (x, y), r,
            color=colors[class_names.index(class_name)],
            thickness=2
        )

        font_scale = min(1,max(3,int(w/500)))
        font_thickness = min(2, max(10,int(w/50)))

        p1, p2 = (int(x - r), int(y - r)), (int(x + r), int(y + r))
        # Text width and height
        tw, th = cv2.getTextSize(
            class_name,
            0, fontScale=font_scale, thickness=font_thickness
        )[0]
        p2 = p1[0] + tw, p1[1] + -th - 10
        cv2.rectangle(
            image,
            p1, p2,
            color=colors[class_names.index(class_name)],
            thickness=-1,
        )
        cv2.putText(image, class_name, (x - r + 1, y - r - 6), cv2.FONT_HERSHEY_SIMPLEX,
            font_scale, (255, 255, 255), font_thickness
        )
    return image


# Function to plot images with the bounding boxes.
def labels_plot(image_paths, label_paths, num_samples, curr_dir):
    """_summary_

    Args:
        image_paths (_type_): _description_
        label_paths (_type_): _description_
        num_samples (_type_): _description_
        curr_dir (_type_): _description_
    """    
    all_training_images = glob.glob(image_paths)
    all_training_labels = glob.glob(label_paths)
    # print(all_training_images)
    all_training_images.sort()
    all_training_labels.sort()

    num_images = len(all_training_images)

    plt.figure(figsize=(12, 6))
    for i in range(num_samples):
        j = random.randint(0,num_images-1)
        image = cv2.imread(all_training_images[j])
        with open(all_training_labels[j], 'r') as f:
            bboxes = []
            labels = []
            label_lines = f.readlines()
            for label_line in label_lines[:20]:
                label = label_line[0]
                bbox_string = label_line[2:]
                x_c, y_c, w, h = bbox_string.split(' ')
                x_c = float(x_c)
                y_c = float(y_c)
                w = float(w)
                h = float(h)
                bboxes.append([x_c, y_c, w, h])
                labels.append(label)
        result_image = plot_circle(image, bboxes, labels)
        plt.subplot(1, 2, i+1)
        plt.imshow(result_image[:, :, ::-1])
        plt.axis('off')
    plt.subplots_adjust(wspace=0)
    plt.tight_layout()
    plt.savefig(f'{curr_dir}/label_plot.png', dpi=600)
    plt.show()
    