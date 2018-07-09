for filename in ./bottom_test/*.jpg; do echo "$filename"; python classify.py "$filename"; done
