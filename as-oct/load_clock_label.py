import pandas as pd

csvpath = csvPath = "I:\octdata\\brightVsDark_label\\eyeId_label_128_all0_add-.csv"
target_df = pd.read_csv (csvpath, index_col='eyeId')  ##### the final label csv File

def readSpecificCol(path):
    df = pd.read_csv (path)
    code = df["Code"]
    # print(list(code))
    list1 = list (code)
    ##### find duplicate id
    # unique_list=[]
    # for i in list1:
    #     if i not in unique_list:
    #         unique_list.append(i)
    #     else:
    #         print(i)
    #
    # #####
    return list1


def generateDf(label_dir):
    df = pd.read_csv (label_dir, index_col='Code', encoding="utf8")
    return df


def orginLabelToRange(strings, numberOfSlices):
    # print("strings", strings)
    if str (strings) == "nan":
        ranges = []
        return ranges
    temp = strings.split (",")  ##
    # temp = "".join(strings.split())
    # temp = temp.split(",")
    ranges = []
    for i in temp:
        if i == "-":
            ranges = []
        elif i == '360':
            ranges = range (12)
        elif "-" in i:  ##当标签为范围表示时候
            temp1 = i.split ("-")
            # for j in temp1:
            #     ranges.append(j)
            start = float (temp1[0])
            # ranges.append(start)
            if start < float (temp1[1]):  ##当标签范围没有跨越12点
                while start < float (temp1[1]):
                    ranges.append (start % 12)
                    start += 1
                    # print(ranges)

                ranges.append (float (temp1[1]) if float (temp1[1]) != 12.5 else 0.5)
            else:  ##当标签范围跨越了12点
                while start < float (temp1[1]) + 12:
                    ranges.append (start % 12)
                    start += 1
                ranges.append (float (temp1[1]) if float (temp1[1]) != 12.5 else 0.5)
        else:  ##当标签是单个钟点时
            try:
                ranges.append (float (i))
            except:
                return []
    # print(ranges, len(ranges))
    return ranges


def generate_new_csv(df):
    # print (df)
    eyeId = df
    # print (eyeId)



def final_label_table(odrange, osrange, eyeid, slices, df): #####
    # print ("start converting to label")
    # print("yeah", df.loc["C2-001-0", "od_left"])
    if (len (osrange) == 12):
        print("360>>>>>>>>>>>>>>>>")
    for i in odrange:
        index_range = clockToIndex(i, slices)
        # print (index_range)
        for index, lr in index_range:
            # print(index, lr)
            print(eyeid+"-"+str(index), "od_"+lr)
            # df.loc[eyeid+"-"+str(index), "od_"+lr] = 1
            df.set_value (eyeid +"-"+ str(index), "od_"+lr, 1)
            # print (df.loc[eyeid + "-" + str(index), "os_" + lr])
    for j in osrange:
        index_range = clockToIndex (j, slices)
        # print(index_range)
        for index, lr in index_range:
            print (eyeid + "-" + str (index), "os_" + lr)
            df.set_value(eyeid + "-" + str (index),  "os_" + lr, 1)
            #[eyeid + "-" + str (index), "os_" + lr] = 1
            # print(df[eyeid + "-" + str (index), "os_" + lr] )
        # i1=i-0.5
        # i2=i+0.5
        # temp = slices * ((3-i)%12 / 6)
        #
        #
        # ##
        # if temp - int (temp) == 0:
        #     print ("index", int (temp))
        #     # df[eyeid + "-" + str (int (temp)), "od_left"] = 1  ####over write specific col
        # else:
        #     print ("index", int (temp))
        #     if int (temp) < 17:
        #         print ("index", int (temp) + 1)


def writeToCsv(eyeId_list, slices):
    eyeId = []
    od_left = []
    od_right = []
    os_left = []
    os_right = []
    for id in eyeId_list:
        for index in range (slices):
            eyeId.append (id + "-" + str (index))
            od_left.append (0)
            od_right.append (0)
            os_left.append (0)
            os_right.append (0)
    dicts = {"eyeId": eyeId, "od_left": od_left, "od_right": od_right, "os_right": os_right, "os_left": os_left}
    dict_df = pd.DataFrame (dicts)
    dict_df.to_csv ("I:\octdata\\brightVsDark_label/eyeId_label_128_all0_add-.csv")

    return 0


def clockToIndex(clock, slices):  ###3点000    九点127
    clock = float(clock)
    flag = ""
    index = []
    index_range = []
    if clock == 360:
        for flag in ["left", "right"]:
            index_range.extend((i, flag) for i in range(slices))
        # print(index_range)
        return index_range

    clock_top = clock-0.5
    clock_bot = clock+0.5

    for i in [clock, clock_bot, clock_top]:
        if i<9 and i>3:
            flag = "left"
            temp = (i+6)%12    ###转化到钟上方
            index.append((slices * ((3 - temp) % 12 / 6), flag))
        else:
            flag = "right"
            temp = i
            index.append((slices * ((3 - temp) % 12 / 6), flag))
    # print(index)   ###[指针自身，下界， 上界]  already transfered to index

    #####根据index 范围返回所有要更改的标签

    if index[0][1]==index[1][1]==index[2][1]:
        # print("ok")
        index_range = [(i, index[0][1]) for i in range(int(index[1][0]), int(index[2][0]))]
    else:
        if slices ==128:
            # print("nay 128")
            if clock ==3:
                index_range= [(i,index[1][1]) for i in range(117,128)]#
                index_range.extend([(i,index[2][1]) for i in range(0, 11)])
            elif clock ==3.5:
                index_range = [(i, "right") for i in range(106, 117)]  #
                index_range.extend ([(i, "left") for i in range (117, 128)])
                # index_range.append((0,"right"))
            elif clock ==8.5:         ####还未改好
                index_range = [(i, "right") for i in range (106, 117)]  #
                index_range.extend ([(i, "left") for i in range (117, 128)])
                # index_range.append((0,"right"))
            elif clock ==9:
                index_range = [(i, "right") for i in range (117, 128)]  #
                index_range.extend ([(i, "left") for i in range (0, 11)])
        elif slices ==18:
            # print ("nay 18")
            if clock == 3:
                index_range = [(i, index[1][1]) for i in range (17, 18)]  #
                index_range.extend ([(i, index[2][1]) for i in range (0, 2)])
            elif clock == 3.5: ###出现一个左右类别不同的
                index_range = [(i, "right") for i in range (0, 1)]  #
                index_range.extend ([(i, "left") for i in range (16, 18)])
                # index_range.append((0,"right"))
            elif clock == 8.5:   ###出现一个左右类别不同的
                index_range = [(i, "right") for i in range (17, 18)]  #
                index_range.extend ([(i, "left") for i in range (1, 3)])
                # index_range.append((0,"right"))
            elif clock == 9:
                index_range = [(i, "right") for i in range (16, 18)]  #
                index_range.extend ([(i, "left") for i in range (0, 1)])

    #####
    # print(index_range)
    return index_range

def overWriteCsv(csvpath, df):

    eyeId_list = readSpecificCol ("I:\\brightVsDarkLabels_modified.csv")  #### unique eyeId
    print(eyeId_list)
    for id in eyeId_list:
        # print (id)
        strings_od = df.loc[id, "Odclock"]
        strings_os = df.loc[id, "Osclock"]
        ranges_od = orginLabelToRange (strings_od, 128)
        ranges_os = orginLabelToRange (strings_os, 128)
        # print ("od range", ranges_od)
        # print ("os range", ranges_os)
        final_label_table (ranges_od, ranges_os, id, 128, target_df)
    # target_df.to_csv("I:\octdata\\brightVsDark_label\\final_eye_label_128_add-.csv")


if __name__ == '__main__':
    label_dir = "I:/brightVsDarkLabels_modified.csv"
    df = generateDf (label_dir)        ##data fram for unique eyeid without index
    # # # strings = df.loc["C2-014", "Osclock"]
    # # # print(strings)
    # # # reallabel(strings, 18)
    # # # print(df)
    # generate_new_csv(df)

    # eyeId_list = readSpecificCol (label_dir)
    # writeToCsv (eyeId_list, 128)
    # for id in eyeId_list:
    #     print(id)
    #     strings_od = df.loc[id, "Osclock"]
    #     strings_os = df.loc[id, "Odclock"]
    #     ranges_od = reallabel(strings_od, 18)
    #     ranges_os = reallabel(strings_os, 18)
    #     print("od range", ranges_od)
    #     print("os range", ranges_os)
    #     # final_label_table(ranges_od,ranges_os,id,)
    #

    overWriteCsv(csvPath, df)
    # clockToIndex("360", 18)