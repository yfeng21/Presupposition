def get_pos(x):
    return x.split('/')[-1]

# keep only text - remove POS
def keep_text(x): # modified this to create the new dataset with words & POS

    word = x.split('/', 1)[0]
    return word

pos_list = ['IN', 'CD', 'NNS', 'FW', 'NNP', 'VBZ', 'VBN', 'JJ', 'NN', 
    'CC', 'RB', 'MD', 'PRP', 'VB', 'RBR', 'VBG', 'POS', 'PRP$', 'JJR', 'DT', 'WDT', 
    'VBD', 'TO', 'VBP', 'RP', 'WP', 'EX', 'NNPS', 'WRB', 'JJS', '-LRB-', 
    '-RRB-', 'RBS', 'WP$', 'PDT', 'UH', 'LS', 'SYM']

def encode_pos(pos):
    pos_list = ['IN', 'CD', 'NNS', 'FW', 'NNP', 'VBZ', 'VBN', 'JJ', 'NN', 
    'CC', 'RB', 'MD', 'PRP', 'VB', 'RBR', 'VBG', 'POS', 'PRP$', 'JJR', 'DT', 'WDT', 
    'VBD', 'TO', 'VBP', 'RP', 'WP', 'EX', 'NNPS', 'WRB', 'JJS', '-LRB-', 
    '-RRB-', 'RBS', 'WP$', 'PDT', 'UH', 'LS', 'SYM']

    if pos == 'DT-':
        pos = 'DT'

    try:
        encoded_pos = pos_list.index(pos)
    except ValueError:
        encoded_pos = -1

    return encoded_pos
    
def read_tags(dataset):

    Y = []
    Z = [[]]
    P = [[]]

    if dataset == 'train':
        datatest_range = range(100,381)+range(400,821)+range(900,2173)
        path_a = './wsj_nodet/train'
    elif dataset == 'valid':
        datatest_range = range(1,100)+range(2400,2455)
        path_a = './wsj_nodet/dev'
    elif dataset == 'test':
        datatest_range = range(2200,2283)+range(2300,2400)
        path_a = './wsj_nodet/test'


    new_file = False


    for i in datatest_range:
        path_b = '/wsj_%04d.mrg.xml.txt' %i
        path = path_a + path_b
        #print "  "
        #print path

        index = 1
        y_index = 1

        end_of_list = False
        fh = open(path)
        for line in fh:
            # print "line (before split)"
            # print line
            # print ""
            X = line.split()
            # print "line", index
            # print X
            # print ""
            #end_of_list = False
            for x in X:
                # print "at beginning of x loop, EoL is:", end_of_list
                # print "x:", x
                u = x
                if "*none" in u:
                    u = u.strip("*none")
                elif "*a" in x:
                    u = u.strip("*a")
                elif "*the" in x:
                    u = u.strip("*the")

                pos = encode_pos(get_pos(u))

                if pos == -1:
                    # print x, "will be discarded. EoL is:", end_of_list
                    continue

                if new_file == True:
                    Z.append([pos]) # new list
                    #P.append([get_pos(x)])

                    if "*none" in x:
                        x = x.strip("*none")
                        Y.append(1)
                        end_of_list = True
                    elif "*a" in x:
                        x = x.strip("*a")                           
                        Y.append(2)
                        end_of_list = True
                    elif "*the" in x:
                        x = x.strip("*the")        
                        Y.append(3)
                        end_of_list = True
                    else:
                        end_of_list = False

                    # print "case 00", "- len(Z):", len(Z), "len(Y)", len(Y)
                    new_file = False
                elif "*none" in x:
                    #print "here 1"
                    x = x.strip("*none")
                    Y.append(1)
                    y_index = y_index + 1
                    if end_of_list == False:
                        Z[-1].append(pos) # append to last list
                        #P[-1].append(get_pos(x))
                        # print "case 1a", "- len(Z):", len(Z), "len(Y)", len(Y)
                    else:
                        Z.append([pos]) # new list
                        #P.append([get_pos(x)])
                        # print "case 1b", "- len(Z):", len(Z), "len(Y)", len(Y)
                    end_of_list = True                
                    # print Z[-1], Y[-1], end_of_list
                    # print " "
                    
                elif "*a" in x:
                    #print "here 2"
                    x = x.strip("*a")                           
                    Y.append(2)
                    y_index = y_index + 1
                    if end_of_list == False:
                        Z[-1].append(pos) # append to last list
                        #P[-1].append(get_pos(x))
                        # print "case 2a", "- len(Z):", len(Z), "len(Y)", len(Y)
                    else:
                        Z.append([pos]) # new list
                        #P.append([get_pos(x)])
                        # print "case 2b", "- len(Z):", len(Z), "len(Y)", len(Y)
                    end_of_list = True                
                    # print Z[-1], Y[-1], end_of_list
                    # print " "

                elif "*the" in x:
                    #print "here 3"
                    x = x.strip("*the")        
                    Y.append(3)
                    y_index = y_index + 1
                    if end_of_list == False:
                        Z[-1].append(pos) # append to last list
                        #P[-1].append(get_pos(x))
                        # print "case 3a", "- len(Z):", len(Z), "len(Y)", len(Y)
                    else:
                        Z.append([pos]) # new list
                        #P.append([get_pos(x)])
                        # print "case 3b", "- len(Z):", len(Z), "len(Y)", len(Y)
                    end_of_list = True 
                    # print Z[-1], Y[-1], end_of_list
                    # print " "
               
                else:
                    #print "here 4"
                    if end_of_list == False:
                        Z[-1].append(pos) # append to last list
                        #P[-1].append(get_pos(x))
                        # print "case 4a", "- len(Z):", len(Z), "len(Y)", len(Y)
                    else:
                        Z.append([pos]) # new list
                        #P.append([get_pos(x)])
                        # print "case 4b", "- len(Z):", len(Z), "len(Y)", len(Y)
                        end_of_list = False
            #print "here 5"
            index = index+1
            # print "len(Z):", len(Z), "len(Y)", len(Y)
            # print " "
            # print " "

        #print "here 6"
        if len(Z)>len(Y):
            #print "here 7"
            Z.pop()
            #P.pop()
            # print "popped."

        new_file = True
        #print "here 8"

    
    return Z


if __name__ == '__main__':
    Z = read_tags('test')
    for zlist in Z[:40]:
        for z in zlist:
            print pos_list[z],
        print