#!python3
# -*- coding: utf-8 -*-
import nbformat
from nbconvert import PythonExporter
from nbformat import v3, v4
from os.path import exists
import shutil
import argparse
import re


parser = argparse.ArgumentParser(
    description='Converts an ipython notebook to a python script, and vice versa.')

parser.add_argument('inputfile', type=str,
                    help='an input file to be converted')
parser.add_argument('outputfile', type=str,
                    help='an output file to convert into')

args = parser.parse_args()


# ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                   --     comment blocks style     --
# ···············································································
cell_header = [
    "## ───────────────────────────────────── ▼ ─────────────────────────────────────\n# {{{",
    "#···············································································"]
cell_footer = "#                                                                            }}}\n## ─────────────────────────────────────────────────────────────────────────────"
#                                                                           }}}
# ────────────────────────────────────────────────────────────────────────────

# ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                    --     notebook -> python     --
# ···············································································


def nbToPy(ifile, ofile):
    original = nbformat.read(ifile, as_version=4)
    source, _ = PythonExporter().from_notebook_node(original)
    text = re.sub('\n# In.*\n\n', cell_footer +
                  "\n\n" + cell_header[0], source)
    text = re.sub(cell_header[0][-5:] + '\n# (.*)', cell_header[0]
                  [-5:] + '              \\1\n' + cell_header[1], text)
    text += cell_footer + "\n"
    text = re.sub(cell_footer, '', text, count=1)
    with open(ofile, "w") as fpout:
        fpout.write(text)
        print("wrote to", ofile)
#                                                                           }}}
# ─────────────────────────────────────────────────────────────────────────────

# ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                    --     python -> notebook     --
# ···············································································


def pyToNb(ifile, ofile):
    fpin = open(ifile, 'r')
    text = fpin.read()
    text = re.sub(cell_header[0][:5] + '.*\n', '', text)
    text = re.sub(cell_header[1][:5] + '.*\n', '', text)
    text = re.sub('## ' + cell_footer[-5:-2] + '.*\n', '', text)
    text = re.sub('\n# *}}}.*\n', '', text)
    text = re.sub('\n# {{{ *', '\n# <codecell>\n# ', text)
    text = re.sub('^#!/.*\n', '', text)
    nbook = v3.reads_py(text)
    nbook = v4.upgrade(nbook)  # Upgrade v3 to v4
    jsonform = v4.writes(nbook) + "\n"
    with open(ofile, "w") as fpout:
        fpout.write(jsonform)
        print("wrote to", ofile)
#                                                                            }}}
# ─────────────────────────────────────────────────────────────────────────────


ifile = args.inputfile
iext = ifile.split('.')[-1]

ofile = args.outputfile
oext = ofile.split('.')[-1]


def makeBackup(f):
    if exists(f):
        shutil.copyfile(f, f + '.backup')


if iext == "ipynb":
    assert(oext == "py")
    print("Translating from notebook to python script")
    makeBackup(ofile)
    nbToPy(ifile, ofile)
elif iext == "py":
    assert(oext == "ipynb")
    print("Translating from python script to notebook")
    makeBackup(ofile)
    pyToNb(ifile, ofile)
else:
    print("Error, file extensions should be ipynb or py")
    exit(1)


