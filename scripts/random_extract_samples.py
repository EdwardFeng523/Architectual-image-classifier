import subprocess
import os
import sys
import tensorflow as tf
import random

flags = tf.app.flags
flags.DEFINE_string('source_dir', '/home/edward/Documents/MLimages/Images_copy', 'Path to the source images for sampling')
flags.DEFINE_string('target_dir0', '/home/edward/Documents/MLimages/jade', 'Path to the sampled images')
flags.DEFINE_string('target_dir1', '/home/edward/Documents/MLimages/crystal', 'Path to the sampled images')
flags.DEFINE_string('target_dir2', '/home/edward/Documents/MLimages/oval', 'Path to the sampled images')
flags.DEFINE_integer('num_sets', 3, 'Number of sets we want the samples to be distributed into')
flags.DEFINE_integer('amount', 60, 'Number of samples we want')
FLAGS = flags.FLAGS


AMOUNT = FLAGS.amount

TARGET_DIRS = [FLAGS.target_dir0, FLAGS.target_dir1, FLAGS.target_dir2]

def move_file(name, idx):
    base_jpg = name + ".jpg"
    complete_dir_jpg = os.path.join(FLAGS.source_dir, base_jpg)
    base_xml = name + ".xml"
    complete_dir_xml = os.path.join(FLAGS.source_dir, base_xml)
    cmdjpg = "cp -i -v " + str(complete_dir_jpg) + " " + str(TARGET_DIRS[idx])
    resjpg = subprocess.check_output(cmdjpg, shell=True)
    print (resjpg)
    cmdxml = "cp -i -v " + str(complete_dir_xml) + " " + str(TARGET_DIRS[idx])
    resxml = subprocess.check_output(cmdxml, shell=True)
    print (resxml)

def iter_sampling(sample_amount):
    for i in range(sample_amount):
        idx = random.randrange(len(dir_lst))
        name_to_move = dir_lst[idx]
        move_file(name_to_move, i % FLAGS.num_sets)
        dir_lst.pop(idx)

dir_lst = []

for infile in os.listdir(FLAGS.source_dir):
    if infile.endswith('.xml'):
        dir_lst.append(os.path.splitext(str(infile))[0])
        print ("scanned", os.path.splitext(str(infile))[0])


if len(dir_lst) > AMOUNT:
    sample_amount = AMOUNT
    iter_sampling(sample_amount)
else:
    sample_amount = len(dir_lst)
    while True:
        print ("The amount of sample you require is greater than or equal to the amount of xml files in the directory. Proceed anyways? (Y/N)")
        input = input()
        if input == "Y" or input == "y":
            iter_sampling(sample_amount)
            break
        elif input == "N" or input == "n":
            sys.exit()






