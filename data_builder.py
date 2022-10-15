import numpy as np
import scipy.io

user_train = '/home/kaykay/Attentive-Group-Recommendation/data/CAMRa2011/userRatingTrain.txt'
user_test = '/home/kaykay/Attentive-Group-Recommendation/data/CAMRa2011/userRatingTest.txt'
group_train = '/home/kaykay/Attentive-Group-Recommendation/data/CAMRa2011/groupRatingTrain.txt'
group_test = '/home/kaykay/Attentive-Group-Recommendation/data/CAMRa2011/groupRatingTest.txt'
file_group_user = '/home/kaykay/Attentive-Group-Recommendation/data/CAMRa2011/groupMember.txt'
M_user = np.zeros((602, 7710))
M_group = np.zeros((290, 7710))
M_group_user = np.zeros((290, 602))
o_group_train = np.zeros((290, 7710))
o_group_test = np.zeros((290, 7710))

def create_M(file_train, file_test, M, flag = False):
    cnt = 0
    with open(file_train) as f:
        line = f.readline()
        while(line!=None and line!=""):
            data = line.split(" ")
            M[int(data[0])][int(data[1])] = int(data[2])
            if flag:
                o_group_train[int(data[0])][int(data[1])] = 1
            line = f.readline()
    f.close()

    with open(file_test) as f:
        line = f.readline()
        while(line!=None and line!=""):
            data = line.split(" ")
            M[int(data[0])][int(data[1])] = int(data[2])
            if flag:
                o_group_test[int(data[0])][int(data[1])] = 1
            line = f.readline()


def create_group_user(file_group_user):
    with open(file_group_user) as f:
        line = f.readline()
        while(line!=None and line!=""):
            group, users = line.split(" ")
            users = users.split(',')
            for user in users:
                M_group_user[int(group)][int(user)] = 1
            line = f.readline()


create_M(user_train, user_test, M_user)
create_M(group_train, group_test, M_group, True)
create_group_user(file_group_user)

M_user_file = '/home/kaykay/GNN/CAMRa2011.mat'
scipy.io.savemat(M_user_file, {'M_user': M_user, 'M_group': M_group, 'M_group_user': M_group_user, 'O_group_train': o_group_train, 'O_group_test': o_group_test})
