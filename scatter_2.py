import numpy as  np
import sys


def between_scatter(data, labels):
    mean_vectors = []
    for label in range(1, 4):
        mean_vectors.append(np.mean(data[labels == label], axis=0))
    # print(mean_vectors)
    tempLen = len(data[0])
    between_scatter = np.zeros((tempLen, tempLen))
    overall_mean = np.mean(data, axis=0)

    for i, mean_vec in enumerate(mean_vectors):
        n = len(data[labels == i + 1, :])
        mean_vec = mean_vec.reshape(tempLen, 1)
        overall_mean = overall_mean.reshape(tempLen, 1)
        between_scatter += n *  (mean_vec - overall_mean).dot((mean_vec - overall_mean).T)
    return between_scatter


def maximize(data, scatter_matrix):
    evals, evecs = np.linalg.eig(scatter_matrix)
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

    data = np.genfromtxt(inputfile_data, delimiter=',')
    data = np.array(data)
    labels = np.genfromtxt(inputfile_label, delimiter=',')
    labels = np.array(labels)

    between_scatter_val = between_scatter(data, labels)
    # print(between_scatter_val)
    final_data, eigenvecs = maximize(data, between_scatter_val)
    np.savetxt(outputFile_vectors, eigenvecs.T, delimiter=',')
    np.savetxt(outputFile_reducedData, final_data, delimiter=",")
