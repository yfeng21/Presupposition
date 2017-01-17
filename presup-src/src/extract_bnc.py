'''
Created on Feb 4, 2016

@author: jcheung
'''

import lxml.etree as ET
import sys, glob, os


def get_raw_text(xml_filename, xsl_filename, out):
    dom = ET.parse(xml_filename)
    xslt = ET.parse(xsl_filename)
    transform = ET.XSLT(xslt)
    newdom = transform(dom)
    out.write(str(newdom))
    

if __name__ == '__main__':
    #xml_filename = './bnc/A0/A01.xml'
    xsl_filename = './justTheWords.xsl'
    
    bnc_texts_dir = sys.argv[1] # where the base Texts directory is 
    out_dir = sys.argv[2] # will recreate the structure
    
    subdirs = glob.glob(os.path.join(sys.argv[1], '*'))
    #print subdirs
    for subdir in subdirs:
        letter = os.path.split(subdir)[1]
        if not os.path.exists(os.path.join(out_dir, letter)):
            os.makedirs(os.path.join(out_dir, letter))
        ssdirs = glob.glob(os.path.join(subdir, '*'))
        
        #print letter, ssdirs
        for ssdir in ssdirs:
            lvl2 = os.path.split(ssdir)[1]
            if not os.path.exists(os.path.join(out_dir, letter, lvl2)):
                os.makedirs(os.path.join(out_dir, letter, lvl2))
            #print lvl2
            fs = glob.glob(os.path.join(ssdir, '*.xml'))
            for f in fs:
                out_fname = os.path.split(f)[1][:-4] + '.txt'
                out_f = os.path.join(out_dir, letter, lvl2, out_fname)
                print f, out_f
                with open(out_f, 'w') as out:
                    get_raw_text(f, xsl_filename, out)