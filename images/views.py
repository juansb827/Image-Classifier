import io
import json
import os

import sys

import time
from django.shortcuts import render
from django.http import HttpResponse

import time
import numpy as np
import tensorflow as tf

from base64 import b64decode
from PIL import Image
from django.core.files.temp import NamedTemporaryFile
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.shortcuts import render

TF_GRAPH = "{base_path}/trained_models/retrained_graph.pb".format(
    base_path=os.path.abspath(os.path.dirname(__file__)))
TF_LABELS = "{base_path}/trained_models/retrained_labels.txt".format(
    base_path=os.path.abspath(os.path.dirname(__file__)))


def load_graph():
    graph = tf.Graph()
    graph_def = tf.GraphDef()

    with open(TF_GRAPH, "rb") as f:
        graph_def.ParseFromString(f.read())
    with graph.as_default():
        tf.import_graph_def(graph_def)

    return graph

def load_labels():
     label = []
     proto_as_ascii_lines = tf.gfile.GFile(TF_LABELS).readlines()
     for l in proto_as_ascii_lines:
        label.append(l.rstrip())
     return label

def read_tensor_from_image_file(image_file, input_height=299, input_width=299,
                input_mean=0, input_std=255):

    file_name = os.path.join(sys.path[0], "images", "daisy.jpg")
    input_name = "file_reader"
    output_name = "normalized"
    file_reader = tf.read_file(file_name, input_name)


    print ("data ", type (tf.gfile.FastGFile(image_file.name, 'rb').read()  ))

    file_reader2= tf.read_file(image_file.name, input_name)

    print(image_file.name)
    print(file_reader)
    print(file_reader2)

    image_reader = tf.image.decode_jpeg(file_reader2, channels=3,
                                            name='jpeg_reader')

    float_caster = tf.cast(image_reader, tf.float32)
    dims_expander = tf.expand_dims(float_caster, 0);
    resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
    normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
    sess = tf.Session()
    result = sess.run(normalized)

    return result



def label_image(image_file):

    file_name = "daisy.jpg"
    file_name = os.path.join(sys.path[0], "images", file_name)

    input_height = 224
    input_width = 224
    input_mean = 128
    input_std = 128
    input_layer = "input"
    output_layer = "final_result"


    t = read_tensor_from_image_file(image_file,
                                    input_height=input_height,
                                    input_width=input_width,
                                    input_mean=input_mean,
                                    input_std=input_std)

    input_name = "import/" + input_layer
    output_name = "import/" + output_layer
    input_operation = graph.get_operation_by_name(input_name);
    output_operation = graph.get_operation_by_name(output_name);

    with tf.Session(graph=graph) as sess:
        start = time.time()
        results = sess.run(output_operation.outputs[0],
                           {input_operation.outputs[0]: t})
        end = time.time()


    results = np.squeeze(results)

    top_k = results.argsort()[-5:][::-1] # last five and reverse the order
    totalTime=format(end - start)

    return top_k,results,totalTime








graph=load_graph()
labels=load_labels()
print ("graph_loaded")

@csrf_exempt
def index(request):
    data = {"success": False}

    if request.method == "POST":
        tmp_f = NamedTemporaryFile()
        noImage=False;
        print ("dsad",request.FILES.get("image"))

        if request.FILES.get("image", None) is not None:
            image_request = request.FILES["image"]
            image_bytes = image_request.read()
            image = Image.open(io.BytesIO(image_bytes))
            image.save(tmp_f, image.format)
        elif request.POST.get("image64", None) is not None:
            base64_data = request.POST.get("image64", None).split(',', 1)[1]
            plain_data = b64decode(base64_data)
            tmp_f.write(plain_data)
        else:
            noImage=True


        if noImage:
            data["error"] = "No image provided"
        else :
            try:
                top_k, results, totalTime=label_image(tmp_f)
                confidences={}
                order= 0;
                for i in top_k:
                    confidences[order]= {"label" : labels[i] , "result" : float(results[i]) }
                    order+=1

                data["success"] = True
                data["confidences"]=confidences;
            except Exception as e:
                s = str(e)
                data["error"] = e


    return JsonResponse(data)
