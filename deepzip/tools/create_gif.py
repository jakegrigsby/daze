import argparse
import glob
import imageio

parser = argparse.ArgumentParser()
parser.add_argument('--filename', '--f', help='output filename', default='cvae.gif')
parser.add_argument('--glob','--g',  help='input file glob', default='image*.png')
args = parser.parse_args()

print('Compiling images from {} to GIF {}.'.format(args.glob, args.filename))

with imageio.get_writer(args.filename, mode='I') as writer:
  filenames = glob.glob(args.glob)
  filenames = sorted(filenames)
  last = -1
  for i,filename in enumerate(filenames):
    frame = 2*(i**0.5)
    if round(frame) > round(last):
      last = frame
    else:
      continue
    image = imageio.imread(filename)
    writer.append_data(image)
  image = imageio.imread(filename)
  writer.append_data(image)