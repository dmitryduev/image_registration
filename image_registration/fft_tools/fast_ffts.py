import numpy as np
import warnings

try:
    import pyfftw
    has_fftw = True

    # mimicking the numpy way: [faster than numpy, but slower than the direct way]
    # def fftwn(array, nthreads=1):
    #     return pyfftw.interfaces.numpy_fft.fftn(array, threads=nthreads)

    # def ifftwn(array, nthreads=1):
    #     return pyfftw.interfaces.numpy_fft.ifftn(array, threads=nthreads)

    # the fastest way:
    def fftwn(array, nthreads=1):
        a = array.astype('complex128')
        b = np.zeros_like(a)
        fft = pyfftw.FFTW(a, b, axes=(0, 1), direction='FFTW_FORWARD', flags=('FFTW_MEASURE',),
                          threads=nthreads, planning_timelimit=None)
        return fft()

    def ifftwn(array, nthreads=1):
        a = array.astype('complex128')
        b = np.zeros_like(a)
        ifft = pyfftw.FFTW(a, b, axes=(0, 1), direction='FFTW_BACKWARD', flags=('FFTW_MEASURE',),
                           threads=nthreads, planning_timelimit=None)
        return ifft()

    # import fftw3
    # has_fftw = True
    #
    # def fftwn(array, nthreads=1):
    #     array = array.astype('complex').copy()
    #     outarray = array.copy()
    #     fft_forward = fftw3.Plan(array, outarray, direction='forward',
    #             flags=['estimate'], nthreads=nthreads)
    #     fft_forward.execute()
    #     return outarray
    #
    # def ifftwn(array, nthreads=1):
    #     array = array.astype('complex').copy()
    #     outarray = array.copy()
    #     fft_backward = fftw3.Plan(array, outarray, direction='backward',
    #             flags=['estimate'], nthreads=nthreads)
    #     fft_backward.execute()
    #     return outarray / np.size(array)
    
except ImportError:
    fftn = np.fft.fftn
    ifftn = np.fft.ifftn
    has_fftw = False
# I performed some fft speed tests and found that scipy is slower than numpy
# http://code.google.com/p/agpy/source/browse/trunk/tests/test_ffts.py However,
# the speed varied on machines - YMMV.  If someone finds that scipy's fft is
# faster, we should add that as an option here... not sure how exactly

__all__ = ['get_ffts']


def get_ffts(nthreads=1, use_numpy_fft=not has_fftw):
    """
    Returns fftn,ifftn using either numpy's fft or fftw
    """
    if has_fftw and not use_numpy_fft:
        def fftn(*args, **kwargs):
            return fftwn(*args, nthreads=nthreads, **kwargs)

        def ifftn(*args, **kwargs):
            return ifftwn(*args, nthreads=nthreads, **kwargs)
    elif use_numpy_fft:
        fftn = np.fft.fftn
        ifftn = np.fft.ifftn
    else:
        # yes, this is redundant, but I feel like there could be a third option...
        fftn = np.fft.fftn
        ifftn = np.fft.ifftn

    return fftn,ifftn
