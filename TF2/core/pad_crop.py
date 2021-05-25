'''
Copyright: 2016-2020 Thomas Kuestner (thomas.kuestner@med.uni-tuebingen.de) under Apache2 license
@author: Thomas Kuestner
'''

import numpy as np

def zpad(x, s, mode='constant'):
    # x: input data
    # s: desired size
    # mode: padding mode (constant, ...)
    if type(s) is not np.ndarray:
        s = np.asarray(s, dtype='f')

    if type(x) is not np.ndarray:
        x = np.asarray(x, dtype='f')

    m = np.asarray(np.shape(x), dtype='f')
    if len(m) < len(s):
        m = [m, np.ones(1, len(s) - len(m))]

    if np.sum(m == s) == len(m):
        return x

    idx = list()
    diff = list()
    for n in range(np.size(s)):

        if np.remainder(s[n], 2) == 0:
            idx.append(np.arange(np.floor(s[n] / 2) + 1 + np.ceil(-m[n] / 2) - 1, np.floor(s[n] / 2) + np.ceil(m[n] / 2)))
        else:
            idx.append(np.arange(np.floor(s[n] / 2) + np.ceil(-m[n] / 2) - 1, np.floor(s[n] / 2) + np.ceil(m[n] / 2) - 1))

        if s[n] != m[n]:
            diff.append( ( int(np.abs(idx[n][0])), int(np.abs(s[n]-1 - idx[n][-1])) ) )
        else:
            diff.append( (0, 0) )

    padval = tuple(i for i in diff)
    return np.pad(x, padval, mode=mode)


def crop(x, s):
    # x: input data
    # s: desired size
    if type(s) is not np.ndarray:
        s = np.asarray(s, dtype='f')

    if type(x) is not np.ndarray:
        x = np.asarray(x, dtype='f')

    m = np.asarray(np.shape(x), dtype='f')
    if len(m) < len(s):
        m = [m, np.ones(1, len(s) - len(m))]

    if np.sum(m == s) == len(m):
        return x

    idx = list()
    for n in range(np.size(s)):
        if np.remainder(s[n], 2) == 0:
            idx.append(list(np.arange(np.floor(m[n] / 2) + 1 + np.ceil(-s[n] / 2) - 1, np.floor(m[n] / 2) + np.ceil(s[n] / 2), dtype=np.int)))
        else:
            idx.append(list(np.arange(np.floor(m[n] / 2) + np.ceil(-s[n] / 2) - 1, np.floor(m[n] / 2) + np.ceil(s[n] / 2) - 1), dtype=np.int))

    index_arrays = np.ix_(*idx)
    return x[index_arrays]

if __name__ == '__main__':
    a = np.ones((64,32,16))
    s = (128,64,32)
    apad = zpad(a,s)
    s = (64,32,16)
    acrop = crop(apad,s)
    import scipy.io as sio
    sio.savemat('/mnt/c/Users/Admin/Desktop/tmp/testpadcrop.mat', {'a': a, 'apad': apad, 'acrop': acrop})


