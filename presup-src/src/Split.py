import cPickle,os

def pickleLoader(pklFile):
    try:
        while True:
            yield cPickle.load(pklFile)
    except EOFError:
        pass


#Split string
def split1(input,output):
    target = open(output, 'ab')
    with open(input,'rb') as f:
         for line in pickleLoader(f):
            context = " ".join(line[1])
            indexOfSpecial = context.rfind("@@@@")
            if indexOfSpecial==-1:
                continue
            else:
                #use place of "." as splitting point
                splitPoint= context.rfind(".", beg=0,end=indexOfSpecial)
                sentence1=context[:splitPoint]
                sentence2=context[splitPoint+1:]
                splitResult=(line[0],sentence1,sentence2)
                cPickle.dump(splitResult,target)
    target.close()

#Split Tokens
def split2(input,output):
    target = open(output, 'wb')
    with open(input,'rb') as f:
         for line in pickleLoader(f):
            context = line[1]
            POS=line[2]
            try:
                indexOfSpecial = context.index('@@@@')
            except ValueError:
                continue
            #use place of "." as splitting point
            subContext=context[:indexOfSpecial]
            try:
                splitPoint= max(loc for loc, val in enumerate(subContext) if val == '.')
            except ValueError:
                continue
            sentence1=context[:splitPoint]
            sentence2=context[splitPoint+1:]
            POS1=POS[:splitPoint]
            POS2=POS[splitPoint+1:]
            splitResult=(line[0],sentence1,sentence2,POS1,POS2)
            cPickle.dump(splitResult,target)
    target.close()

if __name__ == '__main__':
 #    sample = ('still', ['The', 'Old', 'Granary', ',', 'Darwin', 'College', 'This', 'splendid', 'though', 'rather', 'whimsical', 'riverside', 'house', 'was', 'built', 'as', 'a', 'granary', ',', 'conveniently', 'positioned', 'at', 'the', 'very', 'head', 'of', 'the', 'navigable', 'part', 'of', 'the', 'River', 'Cam', 'along', 'which', 'barges', 'laden', 'with', 'corn', 'could', 'travel', 'from', 'King', "'s", 'Lynn', 'and', 'the', 'fenland', 'farming', 'areas', '.', 'Since', 'its', 'conversion', 'to', 'a', 'dwelling', 'house', 'in', 'the', 'late', '19th', 'century', 'its', 'impressive', 'list', 'of', 'distinguished', 'tenants', 'has', '@@@@', 'included', 'Bertrand', 'Russell', 'the', 'mathematician', 'and', 'philosopher', 'and', 'Henry', 'Morris', ',', 'founder', 'of', 'the', 'famous', 'system', 'of', 'Village', 'Colleges', 'in', 'use', 'throughout', 'the', 'county'], ['DT', 'NNP', 'NNP', ',', 'NNP', 'NNP', 'DT', 'JJ', 'IN', 'RB', 'JJ', 'NN', 'NN', 'VBD', 'VBN', 'IN', 'DT', 'NN', ',', 'RB', 'VBN', 'IN', 'DT', 'JJ', 'NN', 'IN', 'DT', 'JJ', 'NN', 'IN', 'DT', 'NNP', 'NNP', 'IN', 'WDT', 'VBZ', 'JJ', 'IN', 'NN', 'MD', 'VB', 'IN', 'NNP', 'POS', 'NNP', 'CC', 'DT', 'NN', 'NN', 'NNS', '.', 'IN', 'PRP$', 'NN', 'TO', 'DT', 'NN',
 # 'NN', 'IN', 'DT', 'JJ', 'JJ', 'NN', 'PRP$', 'JJ', 'NN', 'IN', 'JJ', 'NNS', 'VBZ', '@@@@', 'VBN', 'NNP', 'NNP', 'DT', 'NN', 'CC', 'NN', 'CC', 'NNP', 'NNP', ',', 'NN', 'IN', 'DT', 'JJ', 'NN', 'IN', 'NNP', 'NNP', 'IN', 'NN', 'IN', 'DT', 'NN'])
 #    context=sample[1]
 #    POS=sample[2]
 #    indexOfSpecial = max(loc for loc, val in enumerate(context) if val == '@@@@')
 #    #use place of "." as splitting point
 #    subContext=context[:indexOfSpecial]
 #    splitPoint= max(loc for loc, val in enumerate(subContext) if val == '.')
 #    sentence1=context[:splitPoint]
 #    sentence2=context[splitPoint+1:]
 #    POS1=POS[:splitPoint]
 #    POS2=POS[splitPoint+1:]
 #    splitResult=(sample[0],sentence1,sentence2,POS1,POS2)
 #    print splitResult
 inFile="presup_ptb/test/positive_data.pkl"
 out="presup_ptb/test/positive_data_split.pkl"
 split2(inFile,out)