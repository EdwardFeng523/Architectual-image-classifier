import base64
import sys
import json
img = base64.b64encode(open(sys.argv[1], "rb").read())

with open('request.json', 'w+') as outfile:
    json.dump({"image_bytes": {"b64": img}}, outfile)
