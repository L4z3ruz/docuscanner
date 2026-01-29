import tkinter as tk
from tkinter import filedialog
from PIL import Image

def convert():
    file_path = filedialog.askopenfilename(filetypes=[("JPG Files", "*.jpg")])
    if file_path:
        img = Image.open(file_path)
        save_path = filedialog.asksaveasfilename(defaultextension=".png")
        if save_path:
            img.save(save_path, "PNG")
            label.config(text=f"Saved as {save_path}")

root = tk.Tk()
root.title("JPG → PNG Converter")

btn = tk.Button(root, text="Select JPG", command=convert)
btn.pack(pady=20)

label = tk.Label(root, text="")
label.pack()

root.mainloop()
