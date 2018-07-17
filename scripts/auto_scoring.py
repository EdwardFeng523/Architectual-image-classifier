
# coding: utf-8

# # Imports

import numpy as np
import os
import sys
import json
import io
from lxml import etree as ET
from utils import label_map_util
import tensorflow as tf
from PIL import Image
import logging
from PIL.Image import DecompressionBombWarning
import subprocess

flags = tf.app.flags
flags.DEFINE_string('test_dir', '', 'The directory with all the testing images and their xml annotation files.')
flags.DEFINE_string('model_file', '', 'The path to the protobuf file that represents the model you want to score.')
flags.DEFINE_string('visualize_path', '', 'The directory where you want to put all the visualizations that the machine did wrong on.')
FLAGS = flags.FLAGS

class Result:
    def __init__(self, accurate, true_box):
        self.accurate = accurate
        self.true_box = true_box

class Annotation_Tree:
    def __init__(self, filename):
        self.filename = filename
        self.annotation = ET.Element("annotation")
        ET.SubElement(self.annotation, "filename").text = self.filename

    def add_annotation(self, annotation_name, area):
        object = ET.SubElement(self.annotation, "object")
        ET.SubElement(object, "name").text = annotation_name
        bndbox = ET.SubElement(object, "bndbox")
        ET.SubElement(bndbox, "xmin").text = str(area[0])
        ET.SubElement(bndbox, "ymin").text = str(area[1])
        ET.SubElement(bndbox, "xmax").text = str(area[2])
        ET.SubElement(bndbox, "ymax").text = str(area[3])

    def indent(self, elem, level=0):
        i = "\n" + level*"  "
        if len(elem):
            if not elem.text or not elem.text.strip():
                elem.text = i + "  "
            if not elem.tail or not elem.tail.strip():
                elem.tail = i
            for elem in elem:
                self.indent(elem, level+1)
            if not elem.tail or not elem.tail.strip():
                elem.tail = i
        else:
            if level and (not elem.tail or not elem.tail.strip()):
                elem.tail = i

    def get_tree(self):
        self.indent(self.annotation)
        return ET.ElementTree(self.annotation)

    def write_tree_to_file(self, path):
        self.indent(self.annotation)
        tree = ET.ElementTree(self.annotation)
        dot_idx = self.filename.rfind('.')
        xmlfilename = self.filename[:dot_idx] + '.xml'
        whole_path = os.path.join(path, xmlfilename)
        tree.write(whole_path)

class TFOutput:
    def __init__(self):
        self.sheet_title = None
        self.title_confidence = 0.0
        self.title_label_confidence = 0.0
        self.sheet_num = None
        self.num_confidence = 0.0
        self.num_label_confidence = 0.0
        self.rotation_angle = 0.0
        self.rotation_confidence = 0.0
        self.info_confidence = 0.0
        self.boxes_json = {}
        self.filename = ""
        self.sheet_title_options = {}

# sys.path.append("..")

# ## Object detection imports
# Here are the imports from the object detection module.

source_file = ''
_session = None
_model_name = ''


# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('data', 'label_map.pbtxt')

NUM_CLASSES = 9

# ## Loading label map
# Label maps map indices to category names, so that when our convolution network predicts `5`, we know that this corresponds to `airplane`.  Here we use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

### Helper code

def rotate_box(height, width, xmin, ymin, xmax, ymax, theta):
    if abs(theta) == 0:
        new_xmin = xmin
        new_xmax = xmax
        new_ymin = ymin
        new_ymax = ymax
    if abs(theta) == 270:
        new_xmin = height - ymax
        new_xmax = (ymax - ymin) + new_xmin
        new_ymin = xmin
        new_ymax = xmax
    elif abs(theta) == 180:
        new_xmin = width - xmax
        new_xmax = (xmax - xmin) + new_xmin
        new_ymin = height - ymax
        new_ymax = (ymax - ymin) + new_ymin
    elif abs(theta) == 90:
        new_xmin = ymin
        new_xmax = ymax
        new_ymin = width - xmax
        new_ymax = width - xmin

    return [new_xmin,new_ymin, new_xmax,new_ymax]

def load_image_into_numpy_array(image):
  im_width = image.width
  im_height = image.height
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

def writeBoxesJson(class_idx, box_area):
    json_obj = ''
    if class_idx==1:
        json_obj = {'sheetnum': box_area}
    elif class_idx==2:
        json_obj = {'sheetnum_label': box_area}
    elif class_idx==3:
        json_obj = {'sheettitle': box_area}
    elif class_idx==4:
        json_obj = {'sheettitle_label': box_area}
    elif class_idx==5:
        json_obj = {'info_panel': box_area}
    return json_obj

data = []

def scan_sheet(sess, image_path, xml_path, visualize_path, image_tensor, detection_boxes, detection_scores, detection_classes, num_detections):

    file_name = "hi"

    xml_tree = ET.parse(xml_path)
    xml_root = xml_tree.getroot()

    ### Try opening the image
    try:
        Image.MAX_IMAGE_PIXELS = 1000000000
        image = Image.open(image_path)
    except (DecompressionBombWarning, IOError) as e:
        excType = sys.exc_info()[0].__name__
        if(excType == 'IOError'):
            #file open failed
            logging.error('Image open failed: {}'.format(e))
        elif(excType == 'DecompressionBombWarning'):
            # logging.warning(e)
            logging.error('Decompression bomb attack: {}'.format(e))


    ### Extract the pure file name
    file_name_idx =  image_path.rfind('/')+1
    just_file_name = image_path[file_name_idx:]


    ### Creating a tfoutput
    tf_output = TFOutput()

    tf_output.filename = just_file_name

    ### Tweaking the image

    if image.mode != 'RGB':
       image = image.convert('RGB')

    if image.size[0] > image.size[1]:
        basewidth = 2048
        wpercent = (basewidth/float(image.size[0]))
        hsize = int((float(image.size[1])*float(wpercent)))
        image = image.resize((basewidth,hsize), Image.ANTIALIAS)
    else:
        baseheight = 2048
        hpercent = (baseheight/float(image.size[1]))
        wsize = int((float(image.size[0])*float(hpercent)))
        image = image.resize((baseheight,wsize), Image.ANTIALIAS)

    ### Process image
    process_image(sess=sess, image=image, filename=file_name, image_tensor=image_tensor, detection_boxes=detection_boxes,
                  detection_scores=detection_scores, detection_classes=detection_classes, num_detections=num_detections, tf_output=tf_output)

    if (tf_output.rotation_angle != '0'):
      # rotate image and rescan
      if(tf_output.rotation_angle == '90'):
          image = image.rotate(90, expand=True)
      elif(tf_output.rotation_angle == '180'):
          image = image.rotate(180, expand=True)
      elif(tf_output.rotation_angle == '270'):
          image = image.rotate(270, expand=True)


      logging.info('Rescanning ' + just_file_name + ' due to rotation of {}'.format(tf_output.rotation_angle))
      tf_output = TFOutput()
      process_image(sess=sess, image=image, filename=file_name, image_tensor=image_tensor, detection_boxes=detection_boxes,
                    detection_scores=detection_scores, detection_classes=detection_classes, num_detections=num_detections, tf_output=tf_output)


   ## Do the rotation of machine generated annotations here!

    entry = {'filename': tf_output.filename,
          'sheetnum': tf_output.sheet_num,
          'sheettitle': tf_output.sheet_title,
          'rotationangle': tf_output.rotation_angle,
          'con_infopanel': str(tf_output.info_confidence),
          'con_sheetnum': str(tf_output.num_confidence),
          'con_sheetnum_label': str(tf_output.num_label_confidence),
          'con_sheettitle': str(tf_output.title_confidence),
          'con_sheettitle_label': str(tf_output.title_label_confidence),
          'con_rotation': str(tf_output.rotation_confidence),
          'bounding_boxes': tf_output.boxes_json,
          'modelname': _model_name,
          'sheettitle_options': tf_output.sheet_title_options
          }
    data.append(entry)
    total_count = 0
    correct_count = 0

    print (" ")
    print ("Evaluation for " + just_file_name)
    print (" ")

    newTree = Annotation_Tree(os.path.basename(image_path))
    sheetnumResult = check_accuracy(xml_root, tf_output.boxes_json['sheetnum'], "sheetnum")
    if sheetnumResult.accurate:
        correct_count += 1
    else:
        if sheetnumResult.true_box != [0, 0, 0, 0]:
            newTree.add_annotation("model_predicted_sheetnum", tf_output.boxes_json['sheetnum'])
            newTree.add_annotation("human_labeled_sheetnum", sheetnumResult.true_box)

    if sheetnumResult.true_box == [0, 0, 0, 0]:
        print ("No human labeled sheet_num for this image")
        print (" ")
    else:
        total_count += 1


    sheettitleResult = check_accuracy(xml_root, tf_output.boxes_json['sheettitle'], "sheettitle")
    if sheettitleResult.accurate:
        correct_count += 1
    else:
        if sheettitleResult.true_box != [0, 0, 0, 0]:
            newTree.add_annotation("model_predicted_sheettitle", tf_output.boxes_json['sheettitle'])
            newTree.add_annotation("human_labeled_sheettitle", sheettitleResult.true_box)

    if sheettitleResult.true_box == [0, 0, 0, 0]:
        print ("No human labeled sheet_title for this image")
        print (" ")
    else:
        total_count += 1


    if correct_count != total_count:
        ## We need to write the xml file out.
        newTree.write_tree_to_file(visualize_path)
        cmd = "cp -i " + image_path + " " + visualize_path
        subprocess.call(cmd, shell=True)


    # logging.info("File {} results: {}".format(tf_output.filename, entry))


    return (correct_count, total_count)

def process_image(sess, image, filename, image_tensor, detection_boxes, detection_scores, detection_classes, num_detections, tf_output):

    image_np_expanded = np.expand_dims(image, axis=0)


    # Actual detection
    try:
        (boxes, scores, classes, num) = sess.run(
            [detection_boxes, detection_scores, detection_classes, num_detections],
            feed_dict={image_tensor: image_np_expanded})
    except:
        # detection failed
        logging.error('Detection failed')
        logging.error(sys.exc_info())



    # Visualization of the results of a detection.
    for index,value in enumerate(classes[0]):
        num += 1
        cat_idx = classes[0,index] #refer to label_map.txt for class index mapping
        confidence = scores[0,index]

        # Get bounding box values of sheetnumber areas with confidence 30% or higher
        # if confidence >= 0.3:
        param = {'index': index, 'image': image, 'cat_idx': cat_idx, 'boxes': boxes, 'confidence': confidence,
                 'filename': filename, 'tf_output': tf_output}
        get_results(param)



def get_results(params):
    index = params['index']
    image = params['image']
    cat_idx = params['cat_idx']
    boxes = params['boxes']
    confidence = params['confidence']
    filename = params['filename']
    tf_output = params['tf_output']
    texts = ''

    # Convert bounding box vals to coordinates
    ymin = int(image.size[1]*boxes[0,index][0]) #ymin
    xmin = int(image.size[0]*boxes[0,index][1]) #xmin
    ymax = int(image.size[1]*boxes[0,index][2]) #ymax
    xmax = int(image.size[0]*boxes[0,index][3]) #xmax


    area = rotate_box(image.size[1], image.size[0], xmin, ymin, xmax, ymax, int(tf_output.rotation_angle))
    # area=(xmin,ymin,xmax,ymax)



    # sheetnum
    if (cat_idx == 1) and confidence > tf_output.num_confidence:
        tf_output.boxes_json.update(writeBoxesJson(cat_idx, area))
        tf_output.num_confidence = confidence


    # sheetnum_label
    if (cat_idx == 2) and confidence > tf_output.num_label_confidence:
        tf_output.boxes_json.update(writeBoxesJson(cat_idx, area))
        tf_output.num_label_confidence = confidence


    # sheettitle
    if (cat_idx == 3) and confidence > tf_output.title_confidence:
        tf_output.boxes_json.update(writeBoxesJson(cat_idx, area))
        tf_output.title_confidence = confidence

    # sheettitle_label
    if (cat_idx == 4):  # and confidence > tf_output.title_label_confidence:
        tf_output.boxes_json.update(writeBoxesJson(cat_idx, area))
        tf_output.title_label_confidence = confidence

    #info panel
    if(cat_idx==5):
        tf_output.boxes_json.update(writeBoxesJson(cat_idx, area))
        if(confidence > tf_output.info_confidence):
            tf_output.info_confidence = confidence

    #rotation angle
    if cat_idx==6 or cat_idx==7 or cat_idx==8 or cat_idx==9:
        # print('category_index: {} rotation_angle: {} confidence: {}'.format(cat_idx,category_index[cat_idx]['name'], confidence))
        if(confidence > tf_output.rotation_confidence):
            rotation_angle = category_index[cat_idx]['name']
            found_idx = rotation_angle.find("_")
            if(found_idx != -1):
                found_idx += 1
                tf_output.rotation_angle = rotation_angle[found_idx:]
                tf_output.rotation_confidence = confidence
            tf_output.boxes_json.update(writeBoxesJson(cat_idx, area))

            if tf_output.rotation_angle != '0':
                return

def check_accuracy(xml_root, area, label):
    answer = []
    for child in xml_root:
        if child.tag == "object" and child[0].text == label:
            answer.append(child[-1][0].text)
            answer.append(child[-1][1].text)
            answer.append(child[-1][2].text)
            answer.append(child[-1][3].text)
    if len(answer) == 0:
        return Result(False, [0,0,0,0])


    result = float(area[0]) > float(answer[0]) * 0.95 and float(area[1]) > float(answer[1]) * 0.95 and \
             float(area[2]) < float(answer[2]) * 1.05 and float(area[3]) < float(answer[3]) * 1.05

    print ('class: ' + str(label))
    print ("human labeled box: " + str(answer))
    print ("machine predicted box: " + str(area))
    print ("Succeeded" if result else "Failed")
    print (" ")

    return Result(result, answer)

def main(sourcefile, modelpath, visualize_path):
    global source_file
    source_file = sourcefile
    #
    # global _session
    # _session = session

    logging.getLogger().setLevel(logging.INFO)

    # path to checkpoint file
    PATH_TO_CKPT = modelpath
    model_name_idx =  modelpath.rfind('/')
    modelname = modelpath[:model_name_idx]
    model_name_idx =  modelname.rfind('/')+1
    modelname = modelname[model_name_idx:]

    global _model_name
    _model_name = modelname

    ### Load a (frozen) Tensorflow model into memory.
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

        # Definite input and output Tensors for detection_graph
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        # Each box represents a part of the image where a particular object was detected.
        detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
        detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')

        total_correct = 0
        total_boxes = 0

        with tf.Session(graph=detection_graph) as sess:
            for infile in os.listdir(source_file):
                if infile.endswith('.jpg'):
                    jpgPath = os.path.join(source_file, infile)
                    xmlName = os.path.splitext(str(infile))[0] + ".xml"
                    xmlPath = os.path.join(source_file, xmlName)
                    output = scan_sheet(sess, jpgPath, xmlPath, visualize_path, image_tensor, detection_boxes, detection_scores, detection_classes, num_detections)
                    total_correct += output[0]
                    total_boxes += output[1]

        print (str(total_correct) + "/" + str(total_boxes) + " success")


main(FLAGS.test_dir, FLAGS.model_file, FLAGS.visualize_path)
