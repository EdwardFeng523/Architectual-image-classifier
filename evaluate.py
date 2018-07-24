import subprocess
import matplotlib.pyplot as plt
import numpy as np
import operator
import os
import tensorflow as tf

flags = tf.app.flags
flags.DEFINE_string('source_dir', '/home/edward/Documents/architectual-image-classifier/right_test/', 'Path to the source images for testing')
flags.DEFINE_string('correct_class', 'right', 'The correct class name for these images')
flags.DEFINE_integer('num_classes', 4, 'Number of classes in this classification')
FLAGS = flags.FLAGS


NUM_CLASSES = FLAGS.num_classes

CORRECT_CLASS = FLAGS.correct_class

separated_result = []

for infile in os.listdir(FLAGS.source_dir):
    if infile.endswith('.jpg'):
        complete_dir = os.path.join(FLAGS.source_dir, infile)

        cmd = "python classify.py " + str(complete_dir)

        res = subprocess.check_output(cmd, shell=True)

        print str(infile)
        print res

        separated_result.append(str(infile))
        separated_result.extend(res.splitlines())

plot_dict = {}

bar_color = {}

for i in range(0, len(separated_result), NUM_CLASSES + 1):
    name = separated_result[i]
    for j in range(NUM_CLASSES):
        score_info_lst = separated_result[i + j + 1].split(':')
        if score_info_lst[0] == CORRECT_CLASS:
            plot_dict[name] = float(score_info_lst[1])
            if j == 0:
                bar_color[name] = True
            else:
                bar_color[name] = False

distribution = sorted(plot_dict.items(), key=operator.itemgetter(1))

keys = []

values = []

for tup in distribution:
    keys.insert(0, tup[0])
    values.insert(0, tup[1])


def plot(keys, values, name):

    failure_count = 0

    plt.rcdefaults()

    fig, ax = plt.subplots()

    y_pos = np.arange(0, 3 * len(keys), 3)

    barlist = ax.barh(y_pos, values, align='center',
            color='green')

    for i in range(len(values)):
        if bar_color[keys[i]] == False:
            barlist[i].set_color('r')
            failure_count += 1

    ax.set_yticks(y_pos)
    ax.set_yticklabels(keys)

    ax.invert_yaxis()  # labels read top-to-bottom
    ax.set_title(name)
    figure = plt.gcf()
    figure.set_size_inches(40, 20)
    plt.savefig(str(name), dpi=100)

    print str(len(keys) - failure_count) + "/" + str(len(keys)) + " successful"
    print("Plot successfully saved to ./" + str(name) + ".png")

    plt.clf()

plot(keys, values, CORRECT_CLASS + "_confidence")
