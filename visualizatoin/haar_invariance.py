# Visualize "op then Haar" vs "Haar then op" for both shift and rotation
# We will build single-canvas images (no subplots) by concatenating tiles horizontally.
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage

# Reuse helpers from previous cell: make_horizontal_stripe, haar2d, tile_subbands

def make_horizontal_stripe(h=128, w=128, stripe_h=12, y_center=64):
    img = np.zeros((h,w), dtype=np.float32)
    y0 = max(0, y_center - stripe_h//2)
    y1 = min(h, y_center + stripe_h//2)
    img[y0:y1, :] = 1.0
    return img
def haar_1d_downsample(arr):
    """
    Apply 1D Haar transform along the last axis with downsampling by 2
    Args:
        arr: input array of shape (..., N)
    Returns:
        low: low-frequency components of shape (..., N//2)
        high: high-frequency components of shape (..., N//2)
    """
    # ensure even length along last axis
    N = arr.shape[-1]
    if N % 2 == 1:
        arr = arr[..., :-1]
        N -= 1
    
    # compute pairs along last axis
    a = arr[..., 0::2]  # even indices
    b = arr[..., 1::2]  # odd indices
    
    # Haar basis functions
    low = (a + b) / np.sqrt(2.0)   # approximation (low-pass)
    high = (a - b) / np.sqrt(2.0)  # detail (high-pass)
    
    return low, high
def haar2d(img):
    """
    Single-level separable 2D Haar DWT with downsampling by 2 in both dimensions
    Args:
        img: input image of shape (H, W)
    Returns:
        LL, LH, HL, HH: four subbands each of shape (H//2, W//2)
    """
    H, W = img.shape
    
    # Ensure even dimensions
    if H % 2 == 1:
        img = img[:-1, :]
        H -= 1
    if W % 2 == 1:
        img = img[:, :-1]
        W -= 1

    # Step 1: Apply 1D Haar transform along columns (horizontal filtering)
    # This processes each row independently, reducing width by 2
    low_rows, high_rows = haar_1d_downsample(img)
    # Now low_rows and high_rows have shape (H, W//2)
    
    # Step 2: Apply 1D Haar transform along rows (vertical filtering)
    # We need to transpose to apply along the row dimension
    
    # Process low_rows to get LL and LH
    LL, LH = haar_1d_downsample(low_rows.T)  # Transpose to work along rows
    LL, LH = LL.T, LH.T  # Transpose back to correct orientation
    
    # Process high_rows to get HL and HH  
    HL, HH = haar_1d_downsample(high_rows.T)  # Transpose to work along rows
    HL, HH = HL.T, HH.T  # Transpose back to correct orientation
    
    return LL, LH, HL, HH
def tile_subbands(LL, LH, HL, HH, pad=2):
    # normalize for visualization
    def norm(x):
        x = x.copy()
        if x.size == 0:
            return x
        mn, mx = x.min(), x.max()
        if mx - mn < 1e-9:
            return np.zeros_like(x)
        return (x - mn) / (mx - mn)
    LLn, LHn, HLn, HHn = map(norm, (LL, LH, HL, HH))
    h, w = LL.shape
    canvas = np.ones((2*h + pad, 2*w + pad), dtype=np.float32)
    canvas[0:h, 0:w] = LLn
    canvas[0:h, w+pad: w+pad+w] = LHn
    canvas[h+pad:h+pad+h, 0:w] = HLn
    canvas[h+pad:h+pad+h, w+pad:w+pad+w] = HHn
    return canvas
def coeffs_from_img(img):
    return haar2d(img)

def apply_shift(img, dy=1, dx=0):
    return np.roll(np.roll(img, dy, axis=0), dx, axis=1)

def apply_rotate(img, angle=45):
    return scipy.ndimage.rotate(img, angle, reshape=False, order=1)

def op_then_haar(img, op):
    img2 = op(img)
    return coeffs_from_img(img2)

def haar_then_op(img, op):
    LL, LH, HL, HH = coeffs_from_img(img)
    # apply op to each subband separately
    return (op(LL), op(LH), op(HL), op(HH))

def to_canvas(coeffs):
    LL,LH,HL,HH = coeffs
    return tile_subbands(LL,LH,HL,HH, pad=4)

def concat_h(*imgs, pad=8):
    # imgs are grayscale [H, W]; make a single wide canvas with padding
    h = max(im.shape[0] for im in imgs)
    ws = [im.shape[1] for im in imgs]
    W = sum(ws) + pad*(len(imgs)-1)
    canvas = np.ones((h, W), dtype=np.float32)
    x = 0
    for im in imgs:
        h0,w0 = im.shape
        # top align
        canvas[:h0, x:x+w0] = im
        x += w0 + pad
    return canvas

def coeffs_diff_tile(c1, c2):
    # Build a difference tile (absolute difference) normalized
    LL1,LH1,HL1,HH1 = c1
    LL2,LH2,HL2,HH2 = c2
    def norm_abs(a,b):
        d = np.abs(a-b)
        if d.size == 0: return d
        m = d.max()
        return d if m < 1e-9 else d / m
    return tile_subbands(norm_abs(LL1,LL2), norm_abs(LH1,LH2),
                         norm_abs(HL1,HL2), norm_abs(HH1,HH2), pad=4)

# Base image
img = make_horizontal_stripe()

# SHIFT case
op_shift = lambda im: apply_shift(im, dy=1, dx=0)
c_shift_A = op_then_haar(img, op_shift)   # op → Haar
c_shift_B = haar_then_op(img, op_shift)   # Haar → op
tile_A = to_canvas(c_shift_A)
tile_B = to_canvas(c_shift_B)
tile_D = coeffs_diff_tile(c_shift_A, c_shift_B)
shift_canvas = concat_h(tile_A, tile_B, tile_D, pad=16)

plt.figure(figsize=(12,4))
plt.imshow(shift_canvas, cmap='gray', interpolation='nearest')
plt.axis('off')
plt.title('SHIFT: [op→Haar]   |   [Haar→op]   |   [abs diff per subband]')
plt.show()

# ROTATION case
op_rot = lambda im: apply_rotate(im, angle=45)
c_rot_A = op_then_haar(img, op_rot)
c_rot_B = haar_then_op(img, op_rot)
tile_A = to_canvas(c_rot_A)
tile_B = to_canvas(c_rot_B)
tile_D = coeffs_diff_tile(c_rot_A, c_rot_B)
rot_canvas = concat_h(tile_A, tile_B, tile_D, pad=16)

plt.figure(figsize=(12,4))
plt.imshow(rot_canvas, cmap='gray', interpolation='nearest')
plt.axis('off')
plt.title('ROTATION(45°): [op→Haar]   |   [Haar→op]   |   [abs diff per subband]')
plt.show()
plt.savefig('haar_invariance.png', dpi=300, bbox_inches='tight')
