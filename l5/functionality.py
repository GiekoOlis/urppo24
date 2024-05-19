import matplotlib.pyplot as plt
import numpy as np
import pydicom
import skimage.io
import cv2
import os
import json
from PIL import Image, ImageDraw, ImageColor, ImageFont
from operator import itemgetter

import PIL

corner_points_SD = ["CRV", "CAV", "CRD", "CAD"]
corner_points_FD = ["CRVL", "CAVL", "CRVR", "CAVL"]

palette_color = {
    "C": "#ff0000",
    "T": "#ff7128",
    "L": "#ffcc00",
    "S": "#92d050",
    "F": "#00b0f0"
}

palette_white = {
    "C": "white",
    "T": "white",
    "L": "white",
    "S": "white",
    "F": "white"
}


def read_mrk_json(path_to_markdown, markdown, encoding="utf-8"):
    """
    Read *.mrk.json file containing annotations for dicom file

    Parameters
    ----------
    path_to_markdown : str
        The file location of the *.mrk.json file
    markdown : str
        The file name including ".mrk.json" extension
    encoding : str, optional
        The encoding of *.mrk.json file (default is "utf-8")

    Returns
    -------
    dict
        contains spine element name,
        number of points in annotation,
        list of pairs of coordinates (x, y)
    """

    try:
        with open(os.path.join(path_to_markdown, markdown)) as f:
            data = json.load(f)
    except FileNotFoundError as fnf_error:
        print(fnf_error)
    else:
        control_points = []
        for i in data['markups'][0]['controlPoints']:
            control_points.append({
                'id': i['id'],
                'label': i['label'],
                'position': i['position']
            })
        point_set = {
            'name': markdown.split(".")[0],
            'number_of_points': len(control_points),
            'controlPoints': control_points,
        }
        return point_set


def read_all_markdowns(path_to_labels):
    """
    Read all the annotations for one case (C2-C7, Th1-Th12,
    L1-L5, S1, FH1, FH2)

    Parameters
    ----------
    path_to_labels: str
        Path to folder containing all annotations (26 *.mrk.json files)

    Returns
    -------
    list
        list of dictionaries
    """

    filenames = os.listdir(path_to_labels)
    all_point_sets = []
    for file in filenames:
        all_point_sets.append(read_mrk_json(path_to_labels, file))

    return all_point_sets


def define_circle(p1, p2, p3):
    """
    Returns the center and radius of the circle passing the given 3 points.
    In case the 3 points form a line, returns (None, infinity).
    """
    temp = p2[0] * p2[0] + p2[1] * p2[1]
    bc = (p1[0] * p1[0] + p1[1] * p1[1] - temp) / 2
    cd = (temp - p3[0] * p3[0] - p3[1] * p3[1]) / 2
    det = (p1[0] - p2[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p2[1])

    if abs(det) < 1.0e-6:
        return None, np.inf

    # Center of circle
    cx = (bc*(p2[1] - p3[1]) - cd*(p1[1] - p2[1])) / det
    cy = ((p1[0] - p2[0]) * cd - (p2[0] - p3[0]) * bc) / det

    radius = np.sqrt((cx - p1[0])**2 + (cy - p1[1])**2)
    return (cx, cy), radius


def get_LUT_value(data, window, level):
    """
    Apply Window Width and Window Level parameters to dicom pixel array

    Parameters
    ----------
    data: numpy array
        dicom pixel array
    window: int
        window center
    level: int
        window width

    Returns
    -------
    numpy array
        processed dicom pixel array
    """
    return np.piecewise(
        data,
        [data <= (level - 0.5 - (window - 1) / 2),
         data > (level - 0.5 + (window - 1) / 2)],
        [0, 255, lambda data: ((data - (level - 0.5)) /
                               (window - 1) + 0.5) * (255 - 0)])


# TODO: add window and level parameters as input, if they r not None
# then use input, if they r None try to use from dicom
def get_PIL_image(dataset):
    """Get Image object from Python Imaging Library(PIL)"""
    if ('PixelData' not in dataset):
        raise TypeError("Cannot show image -- DICOM dataset does not have "
                        "pixel data")
    # can only apply LUT if these window info exists
    if ('WindowWidth' not in dataset) or ('WindowCenter' not in dataset):
        # print(dataset.PhotometricInterpretation)
        bits = dataset.BitsAllocated
        samples = dataset.SamplesPerPixel
        if bits == 8 and samples == 1:
            mode = "L"
        elif bits == 8 and samples == 3:
            mode = "RGB"
        elif bits == 16:
            # not sure about this -- PIL source says is 'experimental'
            # and no documentation. Also, should bytes swap depending
            # on endian of file and system??
            mode = "I;16"
        else:
            raise TypeError("Don't know PIL mode for %d BitsAllocated "
                            "and %d SamplesPerPixel" % (bits, samples))

        # PIL size = (width, height)
        size = (dataset.Columns, dataset.Rows)
        print(mode)
        # Recommended to specify all details
        # by http://www.pythonware.com/library/pil/handbook/image.htm
        im = PIL.Image.frombuffer(mode, size, dataset.pixel_array,
                                  "raw", mode, 0, 1)

    else:
        ew = dataset['WindowWidth']
        ec = dataset['WindowCenter']
        ww = int(ew.value[0] if ew.VM > 1 else ew.value)
        wc = int(ec.value[0] if ec.VM > 1 else ec.value)
        image = get_LUT_value(dataset.pixel_array, ww, wc)
        # Convert mode to L since LUT has only 256 values:
        #   http://www.pythonware.com/library/pil/handbook/image.htm
        im = PIL.Image.fromarray(image).convert('L')

    return im


def create_filled_mask(
        path_to_orig_image, all_point_sets, path_to_output,
        colored=False, spacing=None, draw_FH=True):
    output = os.path.join(
        path_to_output,
        path_to_orig_image.split("\\")[-1].split(".")[0]+".png")
    dicom = pydicom.dcmread(path_to_orig_image)
    input = get_PIL_image(dicom)
    if spacing is None:
        spacing = dicom.ImagerPixelSpacing
    width, height = input.size

    if colored is False:
        mask = Image.new('1', (width, height), 'black')
        palette = palette_white
    else:
        mask = Image.new('RGB', (width, height), 'black')
        palette = palette_color
    draw = ImageDraw.Draw(mask)
    for k, markdown in enumerate(all_point_sets):
        if "FH" in markdown['name']:
            if draw_FH:
                coordinates = [(float(i['position'][0]),
                                float(i['position'][1]))
                               for i in markdown['controlPoints']]
                c, r = define_circle(coordinates[0], coordinates[1],
                                     coordinates[2])
                c, r = (c[0] / spacing[0], c[1] / spacing[1]), r / spacing[0]
                draw.point((c[0], c[1]), fill=palette["F"])
                draw.ellipse([(c[0]-r, c[1]-r), (c[0]+r, c[1]+r)],
                             fill=palette["F"], outline='white', width=1)
        else:
            coordinates = [(float(i['position'][0] / spacing[0]),
                            float(i['position'][1]) / spacing[1])
            for i in markdown['controlPoints']]
            draw.polygon(tuple(coordinates),
                         fill=palette[markdown['name'][0]],
                         outline='white', width=1)
            print(coordinates)
    mask.save(output)


labels = read_all_markdowns("/labels/001/001_FD")
create_filled_mask("/images/001/001_FD.dcm", labels, "/labels",
                   colored=True, spacing=None, draw_FH=False)
