import numpy
import argparse
from scipy.spatial.transform import Rotation as R

def convert(list2):
    '''
    convert list of transformation to list of tx ty tz qx qy qz qw
    '''
    list = []
    for l in list2:
        l = [float(i) for i in l]
        r = R.from_matrix([[l[0], l[1], l[2]],
                            [l[4], l[5], l[6]],
                            [l[8], l[9], l[10]]])
        q = r.as_quat()
        list.append((l[3],l[7],l[11],q[0],q[1],q[2],q[3]))
    return list


def read_file_list(file1, file2):
    """
    Reads both the files that are in the format 
    file1 - timestamps
    file2 - T00 T01 T02 T03 T10 T11 T12 T13 T20 T21 T22 T23
    Input:
    file1 -- timestamp.txt
    file2 -- ground_truth.txt
    
    Output:
    list1 -- timestamps
    list2 -- ground_truths
    
    """
    file = open(file1)
    timestamp = file.read()
    data = open(file2)
    groundtruth = data.read()

    lines1 = timestamp.replace(","," ").replace("\t"," ").split("\n") 
    lines2 = groundtruth.replace(","," ").replace("\t"," ").split("\n") 

    list1 = [[v.strip() for v in line.split(" ") if v.strip()!=""] for line in lines1 if len(line)>0 and line[0]!="#"]
    list2 = [[v.strip() for v in line.split(" ") if v.strip()!=""] for line in lines2 if len(line)>0 and line[0]!="#"]

    ## confirm that both have same number of lines
    if len(list1) == len(list2):
        print(" input data time stamps length and ground truth lengths matches ")
    else:
        print("error in input data time stamps length and ground truth length doesn't match")
    

    # list = [(float(l[0]),l[1:]) for l in list if len(l)>1]
    return list1, list2

class myFloat( float ):
    def __str__(self):
        return "%.06f"%self

def write_file(list1, list2):
    '''
    write the file in format timestamps tx ty tz qx qy qz qw
    
    '''
    file1 = open("groundtruth_kitti.txt", 'w')
    for i in range(len(list1)):
        file1.write(str(myFloat(float(list1[i][0]))))
        file1.write(" ")
        for l in list2[i]:
            file1.write(str(myFloat(l)))
            file1.write(" ")
        file1.write("\n")
        # print(str(list2[i]))
    file1.close()
    print("file made \n")
    pass


'''
In this file we have Rotation and translation from the transformation matrix
T00 T01 T02 T03 
T10 T11 T12 T13 
T20 T21 T22 T23
 0   0   0   1
Now we intend to convert this to tx ty tz qx qy qz qw 
 '''
if __name__=="__main__":
    # parse the command line
    parser = argparse.ArgumentParser(description ='''
    this script combines kitti groundtruth file with timestamp and converts to
    timestamp tx ty tz qx qy qz qw ''')
    
    parser.add_argument('first_file',help='time stamps file (format: timestamp)')
    parser.add_argument('second_file',help='ground truth file (format: T00 T01 T02 T03 T10 T11 T12 T13 T20 T21 T22 T23)')
    args = parser.parse_args()
    list1 , list2 = read_file_list(args.first_file, args.second_file)
    converted_list = convert(list2)
    write_file(list1, converted_list)

    