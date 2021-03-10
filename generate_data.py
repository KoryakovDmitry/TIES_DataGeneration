from TFGeneration.GenerateTFRecord import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--filesize', type=int, default=1)  # number of images in a single tfrecord file
parser.add_argument('--threads', type=int, default=1)  # one thread will work on one tfrecord
parser.add_argument('--outpath', default='/Volumes/Seagate/Initflow/PubTabNetDs/ds/')  # directory to store tfrecords

# imagespath,
parser.add_argument('--imagespath', default='/Users/dmitry/Initflow/TIES/ds/UNLV_dataset/unlv_images')
parser.add_argument('--ocrpath', default='/Users/dmitry/Initflow/TIES/ds/UNLV_dataset/unlv_xml_ocr')
parser.add_argument('--tablepath', default='/Users/dmitry/Initflow/TIES/ds/UNLV_dataset/unlv_xml_gt')

parser.add_argument('--visualizeimgs', type=int, default=0)  # if 1, will store the images along with tfrecords
parser.add_argument('--visualizebboxes', type=int, default=0)  # if 1, will store the bbox visualizations in visualizations folder
args = parser.parse_args()

filesize = max(int(args.filesize), 4)
visualizeimgs = False
if (args.visualizeimgs == 1):
    visualizeimgs = True

visualizebboxes = False
if (args.visualizebboxes == 1):
    visualizebboxes = True

distributionfile = 'unlv_distribution'

t = GenerateTFRecord(outpath=args.outpath,
                     filesize=filesize,
                     unlvimagespath=args.imagespath,
                     unlvocrpath=args.ocrpath,
                     unlvtablepath=args.tablepath,
                     visualizeimgs=visualizeimgs,
                     visualizebboxes=visualizebboxes,
                     distributionfilepath=distributionfile
                     )
# t.write_to_tf(args.threads)
t.gen_tf_par(ds_path="/Users/dmitry/Initflow/pubtabnetds/ds_anns_pubnet/anns_1_80000.json", mode="train", max_threads=8)
