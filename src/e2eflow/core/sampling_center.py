import numpy as np

def sampleCenter(nCenter, nKy, nKz):
    r_center = nCenter/100 * np.sqrt(1 + (-1 + (2*(np.floor(nKy/2)-3))/(nKy-1))*(-1 + (2*(np.floor(nKy/2)-3))/(nKy-1)))  # ellipse
    lMaskCenter = np.zeros((nKy,nKz))
    for iL in range(0,nKy):
        for iP in range(0,nKz):
            kyidx = -1 + 2*iL/(nKy-1)  # circle
            kzidx = -1 + 2*iP/(nKz-1)  # circle
            # rcurr = ((-1 + 2*iP/(nKz - 1))*(-1 + 2*iP/(nKz - 1)) + (-1 + 2*iL/(nKy - 1))*(-1 + 2*iL/(nKy - 1)))  # ellipse
            rcurr = np.sqrt(kyidx*kyidx + kzidx*kzidx)  # circle
            if rcurr < r_center:
                lMaskCenter[iL,iP] = 1

    return lMaskCenter

if __name__ == "__main__":
    # percent of size of central ring
    numLin = 256
    numPar = 72
    nCenter = 15  # [%]
    maskcenter = sampleCenter(nCenter, numLin, numPar)

    import matplotlib.pyplot as plt
    plt.imshow(maskcenter)
    plt.show()