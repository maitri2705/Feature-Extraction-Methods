import numpy as np
import sys


def pca2(data):
    data_matrix = data - np.mean(data, 0)
    data_matrix /= np.std(data_matrix, 0)
    data_mul = np.cov(data_matrix.T)
    evals, evecs = np.linalg.eig(data_mul)
    index = np.argsort(evals)[::-1][0:2]
    evecs = evecs[:, index]
    temp_mul = np.matrix(evecs)
    # print(temp_mul)
    final_mat = data_matrix * temp_mul
    # print(final_mat)
    # final_mat[:,1]*=-1
    return final_mat, temp_mul


if __name__ == '__main__':
    arg = sys.argv
    inputfile_data = arg[1]
    inputfile_label = arg[2]
    outputFile_reducedData = arg[4]
    outputFile_vectors = arg[3]

    data = np.genfromtxt(inputfile_data, delimiter=',')
    data = np.matrix(data)
    reducedData, vectors = pca2(data)
    np.savetxt(outputFile_vectors, vectors.T, delimiter=',')
    np.savetxt(outputFile_reducedData, reducedData, delimiter=",")
