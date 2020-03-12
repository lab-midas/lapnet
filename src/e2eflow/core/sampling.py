import os
import numpy as np
import e2eflow.VD_CASPR_CINE as VD_CASPR_CINE


def generate_mask(nSegments=14, acc=15, nRep=1):
        sType = 'CINE'
        sMode = 'interleaved'  # 'interleaved' (CINE), 'noninterleaved' (free-running, CMRA, T2Mapping, ...)
        numLin = 256  # ky points (store it the other way round, for right dim)
        numPar = 72  # kz points
        nRep = nRep # number of time points, should be 1 if you subsample each image individually otherwise it is 4 (resp) or 16 (cardiac)
        #acc = 4  # acceleration
        isVariable = 1  # VDCASPR (=1) or CASPR (=0)
        isGolden = 2  # golden angle increment between spirals (=1), 0 = linear-linear, 1=golden-golden, 2=tinyGolden-golden, 3=linear-golden, 4=noIncr-golden
        isInOut = 1  # spiral in/out sampling => for isGolden=1 & isInOut=1: use tinyGolden-Golden-tinyGolden-Golden-... increments
        isCenter = 0  # sample center point
        isSamePattern = 0  # same sampling pattern per phase/contrast, i.e. no golden/tiny-golden angle increment between them (but still inside the pattern if isGolden==1)
        #nSegments = 10  # #segments = #rings in sampling pattern
        #nRep = 16  # number of repetitions (only free-running)
        nCenter = 15  # percentage of fully sampled center region
        iVerbose = 0  # 0=silent, 1=normal output, 2=all output
        lMask = np.zeros((numPar, numLin))
        kSpacePolar_LinInd = np.zeros((nRep * numLin * numPar, 1))
        kSpacePolar_ParInd = np.zeros((nRep * numLin * numPar, 1))
        out_parameter = np.zeros((3, 1), dtype='float32')
        parameter_list = np.asarray([numLin, numPar, acc, nCenter, nSegments, nRep, isGolden, isVariable, isInOut, isCenter, isSamePattern, iVerbose], dtype='float32')
        res = VD_CASPR_CINE.run(parameter_list, lMask, kSpacePolar_LinInd, kSpacePolar_ParInd, out_parameter)
        n_SamplesInSpace = np.asscalar(out_parameter[0].astype(int))
        nSampled = np.asscalar(out_parameter[1].astype(int))
        nb_spiral = np.asscalar(out_parameter[2].astype(int))
        mask_rep = np.zeros((numPar, numLin, nRep))
        for iRep in range(1, nRep + 1):
            iVec = list()
            for iInner in range(nSegments):
                iVecTmp = [idx - 1 for idx in range((iRep - 1) * nSegments + 1 + iInner, nSampled - nSegments + 1 + iInner + 1, nSegments * nRep)]
                iVec.extend(iVecTmp)
            # iVec = np.asarray(iVec)

            for iI in iVec:
                if (kSpacePolar_LinInd[iI] > 0) and (kSpacePolar_ParInd[iI] > 0):
                    mask_rep[np.asscalar(kSpacePolar_ParInd[iI].astype(int)), np.asscalar(kSpacePolar_LinInd[iI].astype(int)), iRep - 1] += 1

        return mask_rep  # Z x Y x Time


# generate subsampling mask
# mask_rep = generate_mask(15, 4)
# # kspace dimension: X x Y x Z x Time (complex-valued!)
# kspace_sub = np.multiply(kspace, np.expand_dims(mask_rep, axis=0))
# img_sub = np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(kspace_sub, axes=(0, 1, 2)), axes=(0, 1, 2)), axes=(0, 1, 2))