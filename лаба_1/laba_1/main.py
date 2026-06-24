import tkinter as tk
import matplotlib.pyplot as plt
from ImageProcessorApp import ImageProcessorApp

plt.style.use('default')


if __name__ == "__main__":
    root = tk.Tk()
    app = ImageProcessorApp(root)
    root.mainloop()