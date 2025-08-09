# mandelbrot.py
# Simple Mandelbrot set renderer (numpy + matplotlib)

import numpy as np
import matplotlib.pyplot as plt

def mandelbrot(xmin, xmax, ymin, ymax, width, height, max_iter):
    """
    Compute the Mandelbrot escape-time array.
    Returns a 2D array of iteration counts (0..max_iter).
    """
    # coordinate grid
    real = np.linspace(xmin, xmax, width)
    imag = np.linspace(ymin, ymax, height)
    C = real[np.newaxis, :] + 1j * imag[:, np.newaxis]   # (height, width)

    Z = np.zeros_like(C, dtype=np.complex128)
    div_time = np.zeros(C.shape, dtype=int)            
    mask = np.full(C.shape, True, dtype=bool)           

    for i in range(1, max_iter + 1):
        Z[mask] = Z[mask] * Z[mask] + C[mask]           
        escaped = np.abs(Z) > 2.0
        newly_escaped = escaped & mask
        div_time[newly_escaped] = i
        mask &= ~newly_escaped
        if not mask.any():
            break

    div_time[div_time == 0] = max_iter
    return div_time

def plot_mandelbrot(div_time, cmap='hot', extent=None, title=None, save_path=None):
    plt.figure(figsize=(10, 7))
    plt.imshow(div_time, cmap=cmap, extent=extent, origin='lower', interpolation='bilinear')
    plt.colorbar(label='Escape iteration')
    if title:
        plt.title(title)
    plt.xlabel('Re')
    plt.ylabel('Im')
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved to {save_path}")
    plt.show()

if __name__ == "__main__":
    xmin, xmax = -2.0, 1.0
    ymin, ymax = -1.25, 1.25
    width, height = 1200, 1000   
    max_iter = 300

    # Compute
    div_time = mandelbrot(xmin, xmax, ymin, ymax, width, height, max_iter)

    # extent matches coordinate bounds so axes show complex plane
    extent = (xmin, xmax, ymin, ymax)
    plot_mandelbrot(div_time, cmap='magma', extent=extent,
                    title=f"Mandelbrot set ({width}x{height}, iter={max_iter})",
                    save_path="mandelbrot.png")
