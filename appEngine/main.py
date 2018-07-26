#!/usr/bin/env python
from flask import Flask, request
from google.cloud import storage
from online_prediction import predict_json
import base64
import json
import uuid
from PIL import Image

app = Flask(__name__)
client = storage.Client()


@app.route('/', methods=['GET'])
def main_page():
    return "welcome to page"


@app.route('/info-pos/', methods=['GET'])
def get_info_pos():
    print ("got info-pos request")

    uid = str(uuid.uuid4())
    bucket = request.args.get('bucket')
    path = request.args.get('path')

    result = classify("info_panel_classifier", uid, bucket, path)
    res = {'right': result[0], 'bottom': result[1]}
    return json.dumps(res)

@app.route('/image-type/', methods=['GET'])
def get_image_type():
    print ("got image-type request")

    uid = str(uuid.uuid4())
    bucket = request.args.get('bucket')
    path = request.args.get('path')

    result = classify("image_type_classifier", uid, bucket, path)
    res = {'spec': result[0], 'detail': result[1], 'plan': result[2]}
    return json.dumps(res)

@app.errorhandler(500)
def server_error(e):
    logging.exception('An error occurred during a request.')
    return """
    An internal error occurred: <pre>{}</pre>
    See logs for full stacktrace.
    """.format(e), 500


def classify(model, uid, bucket, path):
    bucket = client.get_bucket(bucket)
    blob_img = bucket.blob(path)
    with open('/tmp/' + uid, 'w+') as file_img:
        blob_img.download_to_file(file_img)

    Image.open('/tmp/' + uid).convert('RGB').save('/tmp/' + uid + '.jpg', quality=80)
    # img.save('/tmp/' + uid + '.jpg')
    basewidth = 2048
    with Image.open('/tmp/' + uid + '.jpg') as img:
        # wpercent = (basewidth/float(img.size[0]))
        # hsize = int((float(img.size[1])*float(wpercent)))
        # img = img.resize((basewidth,hsize), Image.LANCZOS)
        # img.save(outfile, "JPEG", quality=100)

        if img.size[0] > img.size[1] and img.size[0] > basewidth:
            print('width: {}'.format(img.size[0]))
            wpercent = (basewidth/float(img.size[0]))
            hsize = int((float(img.size[1])*float(wpercent)))
            img = img.resize((basewidth,hsize), Image.ANTIALIAS)
            img.save('/tmp/' + uid + '.jpg', quality=80)
            print("resized file")
        elif img.size[1] > img.size[0] and img.size[1] > basewidth:
            baseheight = basewidth
            print('height: {}'.format(img.size[1]))
            hpercent = (baseheight/float(img.size[1]))
            wsize = int((float(img.size[0])*float(hpercent)))
            img = img.resize((baseheight,wsize), Image.ANTIALIAS)
            img.save('/tmp/' + uid + '.jpg', quality=80)
            print("resized file")


    img = base64.b64encode(open('/tmp/' + uid + '.jpg', "rb").read())
    result = (predict_json("oasis-build-747", model, {"image_bytes": {"b64": img}}, "V1"))['prediction']
    return result


if __name__ == '__main__':
    app.run(debug=True)
# [END app]
