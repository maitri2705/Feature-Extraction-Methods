import numpy as np
import sys


def reducemin1(data):
    data_matrix = np.matrix(data)
    cov_matrix = np.cov(data_matrix.T)
    evals, evecs = np.linalg.eig(cov_matrix)
    index = np.argsort(evals)[::-1][0:2]
    evecs = evecs[:, index]
    final_mul = np.matrix(evecs)
    # print(final_mul)
    final_ans = data * final_mul
    # print(final_ans)
    return final_ans, final_mul


def reducemin2(reducedData, label, queriedLabel):
    finalReducedData = []
    for i in range(reducedData.shape[0]):
        # print(int(label[i])," ",queriedLabel)
        if int(label[i]) == int(queriedLabel):
            # print(queriedLabel)
            finalReducedData.append(reducedData[i].tolist()[0])
    return finalReducedData


if __name__ == '__main__':
    arg = sys.argv
    inputfile_data = arg[1]
    inputfile_label = arg[2]
    outputFile_reducedData = arg[3]
    outputFile_vectors = arg[4]
    data = np.genfromtxt(inputfile_data, delimiter=',')
    label = np.genfromtxt(inputfile_label, delimiter=',')
    reducedData, vectors = reducemin1(data)
    if len(arg) == 6:
        queriedlabel = arg[5]
        reducedData = reducemin2(reducedData, label, queriedlabel)
        # print(reducedData)
    np.savetxt(outputFile_vectors, vectors, delimiter=',')
    np.savetxt(outputFile_reducedData, reducedData, delimiter=",")
