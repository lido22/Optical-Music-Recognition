from os import walk
import sys

def OMR(filename):
    return " this is a test"
if __name__ == "__main__":
    if sys.argc < 3 :
        print("few number of arguments")
    
    input_folder = sys.argv[1]
    output_folder = sys.argv[2]
    files = []
    for (dirpath, dirnames, filenames) in walk(sys.argv[1]):
        files.extend(filenames)
        break
    for f in files:
        output = OMR(input_folder+f)
        f = open(output_folder+"/"+f[:-3]+".txt", "a")
        f.write(output)
        f.close()