from xml.etree.ElementTree import Element, SubElement, tostring, ElementTree
import argparse
import csv
from xml.dom import minidom
import pandas as pd
import os


def prettify(elem):
    """Return a pretty-printed XML string for the Element.
    """
    rough_string = tostring(elem, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="    ")


parser = argparse.ArgumentParser(description='Translates CSV into XML')
parser.add_argument('inp_file', help='CSV input file')
args = parser.parse_args()

inp_fn = args.inp_file

# fns = []
# im_ws = []
# im_hs = []
# bbox_coords = []
#
# with open(inp_fn) as csvDataFile:
#     next(csvDataFile)
#     csvReader = csv.reader(csvDataFile)
#     for row in csvReader:
#         fns.append(row[0])
#         im_ws.append(row[1])
#         im_hs.append(row[2])
#         bbox_coords.append((row[4:8]))

df = pd.read_csv(inp_fn)
g1 = df.groupby(['filename'])

for _, gr in g1:
    r = gr[0:1]

    top = Element('annotation')

    folder = SubElement(top, 'folder')
    folder.text = 'APHYLLA'
    fn = SubElement(top, 'filename')
    fn.text = str(r['filename'].values[0])

    src = SubElement(top, 'source')
    db = SubElement(src, 'database')
    db.text = 'The APHYLLA Database'
    anno = SubElement(src, 'annotation')
    anno.text = 'Aphylla'
    im_loc = SubElement(src, 'image')
    im_loc.text = 'database'
    fl_id = SubElement(src, 'flickrid')
    fl_id.text = str(r['filename'].values[0][:-4])

    owner = SubElement(top, 'owner')
    fl_id2 = SubElement(owner, 'flickrid')
    fl_id2.text = 'Will_Kuhn'
    name = SubElement(owner, 'name')
    name.text = 'Will Kuhn'

    size = SubElement(top, 'size')
    width = SubElement(size, 'width')
    width.text = str(r['im_w'].values[0])
    height = SubElement(size, 'height')
    height.text = str(r['im_h'].values[0])
    depth = SubElement(size, 'depth')
    depth.text = '3'

    seg = SubElement(top, 'segmented')
    seg.text = '0'

    for i in range(gr['filename'].count()):
        r2 = gr[i:i + 1]

        obj = SubElement(top, 'object')
        name2 = SubElement(obj, 'name')
        name2.text = 'dragonfly'
        pose = SubElement(obj, 'pose')
        pose.text = 'Unspecified'
        trun = SubElement(obj, 'truncated')
        trun.text = '0'
        diff = SubElement(obj, 'difficult')
        diff.text = '0'

        bbox = SubElement(obj, 'bndbox')
        xmin = SubElement(bbox, 'xmin')
        xmin.text = str(float(r2['bbox_l'].values[0]))
        ymin = SubElement(bbox, 'ymin')
        ymin.text = str(float(r2['bbox_t'].values[0]))
        xmax = SubElement(bbox, 'xmax')
        xmax.text = str(float(r2['bbox_r'].values[0]))
        ymax = SubElement(bbox, 'ymax')
        ymax.text = str(float(r2['bbox_b'].values[0]))

    ofn = os.path.join('aphylla', 'xml', str(r['filename'].values[0]) + '.xml')

    # print(prettify(top)[23:-1], '\n\n')

    with open(ofn, 'w') as f:
        f.write(prettify(top)[23:-1])
