import tkinter as tk
from gui import ImageEditorGUI

def main():
    """Главная функция приложения"""
    root = tk.Tk()
    app = ImageEditorGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()