fw=open('new.txt','w')
with open('/mnt/localssd/CUB_200_2011/attributes/image_attribute_labels.txt','r') as fr:
    ls=fr.readlines()
    for l in ls:
        l = l.strip().split()
        if len(l)==5:
            print(l,file=fw)
        else:
            new_l = ' '.join(l[:4])+' '+l[5]
            print(new_l,file=fw)
fw.close()
