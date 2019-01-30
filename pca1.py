import numpy as np
import sys


def pca1_fn(data):
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


if __name__ == '__main__':
    arg = sys.argv
    inputfile_data = arg[1]
    inputfile_label = arg[2]
    outputFile_reducedData = arg[4]
    outputFile_vectors = arg[3]
    # print(inputfile)
    # print(outputfile)
    data = np.genfromtxt(inputfile_data, delimiter=',')
    reducedData, vectors = pca1_fn(data)
    np.savetxt(outputFile_vectors, vectors.T, delimiter=',')
    np.savetxt(outputFile_reducedData, reducedData, delimiter=",")
