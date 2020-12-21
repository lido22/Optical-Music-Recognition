
import argparse
import os
import datetime
# Initialize parser
parser = argparse.ArgumentParser()

parser.add_argument("inputfolder", help = "Input File")
parser.add_argument("outputfolder", help = "Output File")

args = parser.parse_args()

def OMR(filename):
    return " this is a test"

with open(f"{args.outputfolder}/Output.txt", "w") as text_file:
    text_file.write("Input Folder: %s" % args.inputfolder)
    text_file.write("Output Folder: %s" % args.outputfolder)
    text_file.write("Date: %s" % datetime.datetime.now())
files = []
for (dirpath, dirnames, filenames) in os.walk(args.inputfolder):
    files.extend(filenames)
    break
for f in files:
    output = OMR(args.input_folder+"/"+f)
    f = open(args.output_folder+"/"+f[:-3]+".txt", "a")
    f.write(output)
    f.close()


print('Finished !!') 
