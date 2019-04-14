from __future__ import division
from multiprocessing import Pool
import time
import os, sys, struct, math
from PIL import Image as pil_image
import numpy as np
from scipy.io.wavfile import write as wav_write
from numpy.random import seed, randint, rand

FL       =   80   # Lowest  frequency (Hz) in soundscape
FH       =  7600   # Highest frequency (Hz)
FS       = 22050   # Sample  frequency (Hz)
T        =  1.05   # Image to sound conversion time (s)
D        =     1   # Linear|Exponential=0|1 distribution
HIFI     =     1   # 8-bit|16-bit=0|1 sound quality
STEREO   =     0   # Mono|Stereo=0|1 sound selection
DELAY    =     1   # Nodelay|Delay=0|1 model   (STEREO=1)
FADE     =     1   # Relative fade No|Yes=0|1  (STEREO=1)
DIFFR    =     1   # Diffraction No|Yes=0|1    (STEREO=1)
BSPL     =     1   # Rectangular|B-spline=0|1 time window
BW       =     0   # 16|2-level=0|1 gray format in P[][]
CAM      =     1   # Use OpenCV camera input No|Yes=0|1
VIEW     =     1   # Screen view for debugging No|Yes=0|1
CONTRAST =     2   # Contrast enhancement, 0=none



TwoPi = 6.283185307179586476925287
HIST  = (1+HIFI)*(1+STEREO)
WHITE = 1.00
BLACK = 0.00

N = 28
M = 28

# Coefficients used in rnd()
ir = 0
ia = 9301
ic = 49297
im = 233280

TwoPi = 6.283185307179586476925287
HIST  = (1+HIFI)*(1+STEREO)
WHITE = 1.00
BLACK = 0.00

#k     = 0
b     = 0
d     = D
ns    = 2 * int(0.5*FS*T)
m     = int(ns / N)
sso   = 0 if HIFI else 128
ssm   = 32768 if HIFI else 128
scale = 0.5 / math.sqrt(M)
dt    = 1.0 / FS
v     = 340.0                                 # v = speed of sound (m/s)
hs    = 0.20           # hs = characteristic acoustical size of head (m)

def wi(fp,i):
   b0 = int(i%256)
   b1 = int((i-b0)/256)
   fp.write(struct.pack('B',b0 & 0xff))
   fp.write(struct.pack('B',b1 & 0xff))

def wl(fp,l):
   i0 = l%65536
   i1 = (l-i0)/65536
   wi(fp,i0)
   wi(fp,i1)

def rnd():
   global ir, ia, ic, im
   ir = (ir*ia+ic) % im
   return ir / (1.0*im)



def extract_sound_from_voice(img):
    ir = 0
    ia = 9301
    ic = 49297
    im = 233280


    k = 0

    w = [0 for i in range(M)]
    phi0 = [0 for i in range(M)]
    A = [[0 for j in range(N)] for i in range(M)]  # M x N pixel matrix

    # Set lin|exp (0|1) frequency distribution and random initial phase
    if (d):
        for i in range(0, M): w[i] = TwoPi * FL * pow(1.0 * FH / FL, 1.0 * i / (M - 1))
    else:
        for i in range(0, M): w[i] = TwoPi * FL + TwoPi * (FH - FL) * i / (M - 1)

    for i in range(0, M):
        ir = (ir * ia + ic) % im
        rndd = ir / (1.0 * im)
        phi0[i] = TwoPi * rndd


    gray = np.asarray(img)
    gray.flags.writeable = True


    imgs_reverse = np.flip(img, axis=0)
    avg = np.mean(img, keepdims=True)

    px = imgs_reverse + CONTRAST * (imgs_reverse - avg)
    px = np.maximum(px, 0.0)
    px = np.minimum(px, 255.0)

    A = np.where(np.equal(px, 0), px, pow(10.0, (px / 16 - 15) / 10.0) )


    # Write 8/16-bit mono/stereo .wav file
    tau1 = 0.5 / w[M - 1]
    tau2 = 0.25 * (tau1 * tau1)
    y = yl = yr = z = zl = zr = 0.0

    # expanding A
    num = int(np.floor(ns / N))
    B = np.reshape(A, [M, N, 1])
    A_expand = np.reshape(np.tile(B, [1, 1, 1, num]), [M, N * num])
    A_expand_revese = A_expand[0:, N*num-(ns - N*num):N*num]
    A_expand = np.concatenate((A_expand, A_expand_revese), axis=1)
    A_expand = np.transpose(A_expand, axes=[1, 0])

    frames = np.arange(ns, dtype='float32')
    frames = np.tile(np.reshape(frames, [ns, 1]), [1, M])
    frames_dt = frames * dt

    w = np.expand_dims(w, axis=0)
    phi0 = np.expand_dims(phi0, axis=0)
    ss = np.sum(A_expand * np.sin(w * frames_dt + phi0), axis=-1)

    frames = []
    while k < ns and not STEREO:
        y = ss[k]
        l = sso + 0.5 + scale * ssm * y 

        if l >= sso - 1 + ssm: l = sso - 1 + ssm
        if l < sso - ssm: l = sso - ssm
        frames.append(l)
        k = k + 1

    return frames



if __name__ == '__main__':

    seed(1)
    #imgs = rand(28,28,1)
    imgs = randint(low=0, high=255, size=(20, 28, 28, 1))

    '''
    img = pil_image.open('/mount/hudi/voice/image/cifar10/test/72_ship.png')
    if img.mode != 'L':
        gray = img.convert('L')
    width_height_tuple = (M, N)
    if img.size != width_height_tuple:
        gray = gray.resize(width_height_tuple, pil_image.NEAREST)

    avg = 0.0
    for i in range(M):
        for j in range(N):
            avg += gray[M - 1 - i, j]
    '''

    px_mat = np.zeros(shape=(20,ns))
    for idx in range(20):
        img = imgs[idx, :, :, :]
        outputs = extract_sound_from_voice(img)
        px_mat[idx,:] = np.asarray(outputs)

    print('test.')
