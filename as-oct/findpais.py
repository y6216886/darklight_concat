from collections import defaultdict

def generateLight(path):
    list1 = []
    list2 = []
    for line in open(path):
        lightType = line.split ("_")[0].split ("-")[-1]
        eyeType = line.split ("_")[1]
        if lightType == "L":
            # print(line, eyeType)
            list1.append(line)
        if lightType == "D":
            list2.append(line)
    print(len(list2), len(list1))
    # list = set(list)
    # print(list)
    return list1, list2

def generateit1(path):
    list1 = []
    for line in open(path):
        eyeType = line.split ("_")[3]
        list1.append(line)
    # list = set(list)
    # print(list)
    return list1

def generatedict(list1):
    dicts = defaultdict(list)
    for line in list1:

        ###lightType = line.split ("_")[0].split ("-")[-1]
        ###eyeType = line.split ("_")[1]

        personID = line.split ("-")[0] + "-" + line.split ("-")[1]
        lightType =line.split ("_")[0].split ("-")[-1]
        index = line.split ("_")[-1].strip (".jpg\n")
        eyeType = eyeType = line.split ("_")[1]
        key = personID  +eyeType+ lightType
        # print(key)
        dicts[key].append(line)
    print(dicts)
    return dicts
def findpair(list1, list2, dicts):   ## light_lis, dark_lis, dicts
    num=0
    pairlist_dark=[]
    pairlist_light=[]
    for line in list2:
        personID = line.split ("-")[0]+"-"+line.split("-")[1]
        lightType = line.split ("-")[2].split ("_")[0]
        index = line.split("_")[-1].strip(".jpg\n")
        eyeType = line.split ("_")[1]
        key = personID  +eyeType+ "L"
        if dicts[key] ==[]:
            print("pairs not found")
            print(line)
            num+=1
        else:
            pairlist_dark.append(line)
            pairlist_light.append(dicts[key][int(index)])
        print(num)
    f1= open("I:/octdata/pairlist_dark_128.txt", "a")
    f2 = open ("I:/octdata/pairlist_light_128.txt", "a")
    for line in pairlist_dark:
        f1.writelines(line)
    for line in pairlist_light:
        f2.writelines(line)

    return pairlist_dark, pairlist_light

def checkpair(pairlist_dark, pairlist_light):
    for i in range(len(pairlist_light)):
        line = pairlist_dark[i]

        personID = line.split ("-")[0] + "-" + line.split ("-")[1]
        lightType = line.split ("-")[2].split ("_")[0]
        index = line.split ("_")[-1].strip (".jpg\n")
        eyeType = line.split ("_")[3]
        key_dark = personID + eyeType + index + lightType

        line = pairlist_light[i]
        line1 = line
        personID = line.split ("-")[0] + "-" + line.split ("-")[1]
        lightType = line.split ("-")[2].split ("_")[0]
        index = line.split ("_")[-1].strip (".jpg\n")
        eyeType = line.split ("_")[3]
        key_light = personID + eyeType + index + lightType

        if key_dark[-1]=="D" and key_light[-1]=="L" and key_dark[:-1]==key_light[:-1]:
            # print ("ok")
            continue

        else:
            print(line1, line)
if __name__ == '__main__':
    light_lis, dark_lis = generateLight ("I:\octdata/128SlicesFilename.txt")
    print(len(light_lis), len(dark_lis))
    list1= generateit1("I:\octdata/128SlicesFilename.txt")
    dicts = generatedict(list1)
    pairlist_dark, pairlist_light = findpair(light_lis, dark_lis, dicts)
    checkpair(pairlist_dark, pairlist_light)