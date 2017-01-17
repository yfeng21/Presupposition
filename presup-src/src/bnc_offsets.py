'''
Created on Mar 3, 2016

@author: jcheung
'''

import sys, glob, os

if __name__ == '__main__':
    bnc_texts_dir = sys.argv[1]
    out_dir = sys.argv[2]
    
    subdirs = glob.glob(os.path.join(bnc_texts_dir, '*'))
    for subdir in subdirs:
        letter = os.path.split(subdir)[1]
        ssdirs = glob.glob(os.path.join(subdir, '*'))
        
        for ssdir in ssdirs:
            lvl2 = os.path.split(ssdir)[1]
            
            #print lvl2
            fs = glob.glob(os.path.join(ssdir, '*.xml'))
            for f in fs:
                my_name = os.path.split(f)[1]
                my_name = os.path.join(out_dir, my_name + '.offset')
                print my_name
                with open(my_name, 'w') as out:
                    s = open(f).read()
                    
                    blocks = s.split('<div level="1"')
                    
                    i = 1
                    for block in blocks[1:]:
                        rel = block.split('>', 1)[1]
                        if len(rel.split('<s n="', 1)) < 2:
                            continue
                        snum = rel.split('<s n="', 1)[1]
                        snum_final = snum.split('"', 1)[0]
                        
                        
                        out.write('%d %s\n' % (i, snum_final))
                        i += 1