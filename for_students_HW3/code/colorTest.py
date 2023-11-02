import tkinter as tk
from tkinter import ttk

# Create the main window
root = tk.Tk()
root.title("Scrollable Grid")

# Create a frame to hold the grid
frame = ttk.Frame(root, padding="3 3 12 12")
frame.grid(column=0, row=0, sticky=(tk.N, tk.W, tk.E, tk.S))
frame.columnconfigure(0, weight=1)
frame.rowconfigure(0, weight=1)

# Create a scrollable canvas to hold the grid
canvas = tk.Canvas(frame)
canvas.grid(column=0, row=0, sticky=(tk.N, tk.W, tk.E, tk.S))
scrollbar = ttk.Scrollbar(frame, orient="vertical", command=canvas.yview)
scrollbar.grid(column=1, row=0, sticky=(tk.N, tk.S))
canvas.configure(yscrollcommand=scrollbar.set)
canvas.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

# Create the grid
grid = ttk.Frame(canvas)
canvas.create_window((0, 0), window=grid, anchor="nw")
for i in range(20):
    for j in range(5):
        checkbox = ttk.Checkbutton(grid, text=f"Option {i*5+j}")
        checkbox.grid(column=j, row=i, sticky=tk.W)

# Add padding to the grid
for child in grid.winfo_children():
    child.grid_configure(padx=5, pady=5)

# Run the main event loop
root.mainloop()
