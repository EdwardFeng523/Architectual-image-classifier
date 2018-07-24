#!/usr/bin/env python
from flask import Flask, request
from google.cloud import storage
from online_prediction import predict_json
import base64
from PIL import Image

app = Flask(__name__)
client = storage.Client()


@app.route('/', methods=['GET'])
def main_page():
    return "welcome to page"


@app.route('/info-pos/', methods=['GET'])
def get_info_pos():
    # separator = word.find('/')
    print ("got info-pos request")
    # json_str = request.get_json()
    bucket = client.get_bucket(request.args.get('bucket'))
    blob_img = bucket.blob(request.args.get('path'))
    with open('/tmp/my-secure-file', 'w+') as file_img:
        blob_img.download_to_file(file_img)
    img = Image.open('/tmp/my-secure-file')
    img.save('/tmp/my-secure-file.jpg')



    basewidth = 2048
    with Image.open('/tmp/my-secure-file.jpg') as img:
        # wpercent = (basewidth/float(img.size[0]))
        # hsize = int((float(img.size[1])*float(wpercent)))
        # img = img.resize((basewidth,hsize), Image.LANCZOS)
        # img.save(outfile, "JPEG", quality=100)

        if img.size[0] > img.size[1] and img.size[0] > basewidth:
            print('width: {}'.format(img.size[0]))
            wpercent = (basewidth/float(img.size[0]))
            hsize = int((float(img.size[1])*float(wpercent)))
            img = img.resize((basewidth,hsize), Image.ANTIALIAS)
            img.save('/tmp/my-secure-file.jpg', quality=100)
            print("resized file")
        elif img.size[1] > img.size[0] and img.size[1] > basewidth:
            baseheight = basewidth
            print('height: {}'.format(img.size[1]))
            hpercent = (baseheight/float(img.size[1]))
            wsize = int((float(img.size[0])*float(hpercent)))
            img = img.resize((baseheight,wsize), Image.ANTIALIAS)
            img.save('/tmp/my-secure-file.jpg', quality=100)
            print("resized file")



    img = base64.b64encode(open('/tmp/my-secure-file.jpg', "rb").read())
    result = (predict_json("oasis-build-747", "info_panel_classifier", {"image_bytes": {"b64": img}}, "V1"))['prediction']
    res = {'right': result[0], 'bottom': result[1]}
    return str(res)

@app.route('/image-type/', methods=['GET'])
def get_image_type():
    # separator = word.find('/')
    print ("got image-type request")
    # json_str = request.get_json()
    bucket = client.get_bucket(request.args.get('bucket'))
    blob_img = bucket.blob(request.args.get('path'))
    with open('/tmp/my-secure-file', 'w+') as file_img:
        blob_img.download_to_file(file_img)
    img = base64.b64encode(open('/tmp/my-secure-file', "rb").read())
    result = (predict_json("oasis-build-747", "image_type_classifier", {"image_bytes": {"b64": img}}, "V1"))['prediction']
    res = {'spec': result[0], 'detail': result[1], 'plan': result[2]}
    return str(res)

@app.errorhandler(500)
def server_error(e):
    logging.exception('An error occurred during a request.')
    return """
    An internal error occurred: <pre>{}</pre>
    See logs for full stacktrace.
    """.format(e), 500


def resize(path):
    basewidth = 2048
    with Image.open(path) as img:
        # wpercent = (basewidth/float(img.size[0]))
        # hsize = int((float(img.size[1])*float(wpercent)))
        # img = img.resize((basewidth,hsize), Image.LANCZOS)
        # img.save(outfile, "JPEG", quality=100)

        if img.size[0] > img.size[1] and img.size[0] > basewidth:
            print('width: {}'.format(img.size[0]))
            wpercent = (basewidth/float(img.size[0]))
            hsize = int((float(img.size[1])*float(wpercent)))
            img = img.resize((basewidth,hsize), Image.ANTIALIAS)
            img.save(path, quality=100)
            print("resized file {}".format(outfile))
        elif img.size[1] > img.size[0] and img.size[1] > basewidth:
            baseheight = basewidth
            print('height: {}'.format(img.size[1]))
            hpercent = (baseheight/float(img.size[1]))
            wsize = int((float(img.size[0])*float(hpercent)))
            img = img.resize((baseheight,wsize), Image.ANTIALIAS)
            img.save(path, quality=100)
            print("resized file {}".format(outfile))


if __name__ == '__main__':
    app.run(debug=True)
# [END app]
