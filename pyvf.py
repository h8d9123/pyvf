import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.signal as sig
import scipy.optimize as opt

def getResidues(ss, yy, p0, asymptote='none'):
    cMat1 = np.transpose(np.array([1 / (ss-pole) for pole in p0]))
    cMat2 = np.transpose(np.array([-yy / (ss-pole) for pole in p0]))
    if asymptote=='none':
        cMat = np.concatenate([cMat1, cMat2], axis=1)
    if asymptote=='constant':
        cMat = np.concatenate([cMat1, np.transpose([np.ones(len(ss))]), cMat2], axis=1)
    if asymptote=='linear':
        cMat = np.concatenate([cMat1, np.transpose([np.ones(len(ss))]), np.transpose([ss]), cMat2], axis=1)
    yyReal = np.real(yy)
    yyImag = np.imag(yy)
    cMatReal = np.real(cMat)
    cMatImag = np.imag(cMat)
    rMat = np.concatenate([np.concatenate([cMatReal, cMatImag]),
                           np.concatenate([-cMatImag, cMatReal])],
                          axis=1)
    ryy = np.concatenate([yyReal, yyImag])
    weight = 1/np.abs(yy)**2
    #weight = 1/(np.log10(np.abs(yy)+1))**2
    weight = np.tile(weight, 2)
    weight = np.diag(weight)
    rMatWeight = np.squeeze(np.array(np.matrix(rMat).T * np.matrix(weight) * np.matrix(rMat)))
    ryyWeight = np.squeeze(np.array(np.matrix(rMat).T * np.matrix(weight) * np.matrix(ryy).T))
    qFit = np.linalg.lstsq(rMatWeight, ryyWeight)
    print(qFit)
    xFit = qFit[0]
    xFit = xFit.reshape(2, len(xFit)/2)
    xFit = xFit[0] + 1j*xFit[1]
    yy_res = xFit[:len(p0)]
    sigma_res = xFit[-len(p0):]
    asympArr = xFit[len(p0):-len(p0)]
    """
    if asymptote=='none':
        asympArr = []
    if asymptote=='constant':
        asympArr = [xFit[len(p0)]]
        sigma_res = xFit[-len(p0):]
    if asymptote=='linear':
        asympArr = xFit[len(p0):len(p0+2)]
        sigma_res = xFit[-len(p0):]
    """
    return yy_res, asympArr, sigma_res

def vectFitSingle(ss, yy, p0, asymptote='none'):
    _, _, sigma_res = getResidues(ss, yy, p0, asymptote)
    ones_col = np.transpose(np.matrix(np.ones(len(sigma_res))))
    sigma_res_row = np.matrix(sigma_res)
    Hmat = np.matrix(np.diag(p0)) - ones_col*sigma_res_row
    yy_poles, _ = np.linalg.eig(Hmat)
    yy_res, asympArr, _ = getResidues(ss, yy, yy_poles, asymptote)

    yy_num, yy_den = sig.invres(yy_res, yy_poles, np.real(asympArr))
    return sig.lti(yy_num, yy_den)

def vectFit(ss, yy, p0=[], numIter=10, asymptote='none', makePlot=True):
    """Performs vector fitting of complex data and returns a scipy.signal.lti model.
    
    Parameters
    ----------
    ss : array of s-parameters (usually 2*np.pi*1j*ff, where ff is the non-angular Fourier frequency)
    yy : array of complex values to be fitted
    p0 : array of guesses for poles
    numIter : number of times to iterate the vector fitting algorithm
    asymptote : 'none', 'constant', or 'linear'. If 'none', the underlying function is assumed to
        be given entirely in terms of residues and poles. If 'constant', a complex constant d will
        be added. If 'linear', a complex linear term d + ss*e will be added.
    makePlot : generate a Bode plot (with residuals) of the data and the fit.

    Returns
    -------
    yyLti : scipy.signal.lti representation of the vector-fitted data
    figHandle : if makePlot is true, vectFit also returns a handle to the Bode plot figure.

    """
    thisGuess = p0
    for ii in range(numIter):
        _, _, sigma_res = getResidues(ss, yy, thisGuess, asymptote)
        ones_col = np.transpose(np.matrix(np.ones(len(sigma_res))))
        sigma_res_row = np.matrix(sigma_res)
        Hmat = np.matrix(np.diag(thisGuess)) - ones_col*sigma_res_row
        yy_poles, _ = np.linalg.eig(Hmat)
        yy_res, asympArr, _ = getResidues(ss, yy, yy_poles, asymptote)
        print(asympArr)
        yy_num, yy_den = sig.invres(yy_res, yy_poles, asympArr)
        yyLti = sig.lti(yy_num, yy_den)
        # The following loop flips poles as necessary to keep them
        # in the left-hand plane
        for poleIndex, pole in enumerate(yyLti.poles):
            if np.real(pole) > 0:
                yyLti.poles[poleIndex] = -np.conjugate(pole)
        thisGuess = yyLti.poles
    if makePlot==True:
        figHandle = bodePlot(ss, yy, yyLti)
        return yyLti, figHandle
    return yyLti

def bodePlot(ss, yy, theLti):
    # Use gridspec to set up the aspect ratio of the Bode plot
    gs = mpl.gridspec.GridSpec(4, 1, height_ratios=(3, 1, 3, 1))
    hPlot = plt.figure(figsize=(8, 14))
    axMag = hPlot.add_subplot(gs[0])
    axPha = hPlot.add_subplot(gs[2], sharex=axMag)
    axResMag = hPlot.add_subplot(gs[1], sharex=axMag)
    axResPha = hPlot.add_subplot(gs[3], sharex=axPha)
    _, fitMag, fitPha = theLti.bode(ss/1j)
    fitComplex = 10**(fitMag/20) * np.exp(1j*fitPha/180*np.pi)
    ff = ss/(2j*np.pi)
    axMag.loglog(ff, np.abs(yy), label='Original', alpha=0.7, lw=2)
    axMag.loglog(ff, np.abs(fitComplex), label='Fitted', alpha=0.7, lw=2)
    axMag.set_ylim(0.5*np.min(np.abs(yy)), 2*np.max(np.abs(yy)))
    axMag.grid(linestyle='solid', which='major')
    axMag.legend(loc='best', framealpha=0.7)
    axMag.set_ylabel('Amplitude')
    axPha.semilogx(ff, np.angle(yy, deg=True), alpha=0.7, lw=2)
    axPha.semilogx(ff, np.angle(fitComplex, deg=True), alpha=0.7, lw=2)
    axPha.set_yticks(np.arange(-270, 270, 45))
    axPha.set_yticks(np.arange(-270, 270, 15), minor=True)
    axPha.set_ylim(-190, 190)
    axPha.grid(linestyle='solid', which='major')
    axPha.set_ylabel('Phase (deg.)')
    # Now compute and plot the residual
    resComplex = fitComplex - yy
    # For the magnitude, we plot the *fractional* residual
    axResMag.loglog(ff, np.abs(resComplex)/np.abs(fitComplex), alpha=0.7, lw=2, c=(0, 0.6, 0))
    axResMag.set_ylim(0.5*np.min(np.abs(resComplex)/np.abs(fitComplex)), 2*np.max(np.abs(resComplex)/np.abs(fitComplex)))
    axResMag.grid(linestyle='solid', which='major')
    axResMag.set_ylabel('Frac. res. amplitude')
    resAngle = (np.unwrap(np.angle(fitComplex)) - np.unwrap(np.angle(yy))) * 180 / np.pi
    axResPha.semilogx(ff, resAngle,
                      alpha=0.7, lw=2, c=(0, 0.6, 0))
    axResPha.set_ylim(-1.1*np.max(np.abs(resAngle)),
                   1.1*np.max(np.abs(resAngle)))
    axResPha.grid(linestyle='solid', which='major')
    axResPha.set_ylabel('Residual phase')
    axResPha.set_xlabel('Frequency (Hz)')
    hPlot.tight_layout()
    hPlot.subplots_adjust(hspace=0.2)
    hPlot.show()
    return hPlot
