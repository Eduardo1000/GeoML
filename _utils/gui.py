import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import glob
import pandas as pd


class App(tk.Frame):
    def __init__(self):
        super().__init__()
        self.master.title('Classifier')

        self._dir = None

        self.num_pages = 0
        self._curr_page = 0

        self.df = None

        fram = tk.Frame(self)
        tk.Button(fram, text="Open Folder", command=self.open).pack(side=tk.LEFT)
        tk.Button(fram, text="Save", command=self.save).pack(side=tk.LEFT)
        tk.Button(fram, text="Prev", command=self.prev).pack(side=tk.LEFT)
        tk.Button(fram, text="Next", command=self.next).pack(side=tk.LEFT)
        fram.pack(side=tk.TOP, fill=tk.BOTH)

        self.la = tk.Label(self)
        self.la.pack()

        fram2 = tk.Frame(self)
        tk.Button(fram2, text="good", command=lambda: self.set_label('good')).pack(side=tk.LEFT)
        tk.Button(fram2, text="bad", command=lambda: self.set_label('bad')).pack(side=tk.LEFT)
        tk.Button(fram2, text="ugly", command=lambda: self.set_label('ugly')).pack(side=tk.LEFT)

        fram2.pack(side=tk.BOTTOM)

        self.pack()

    def set_label(self, x):
        if self.df is not None:
            self.df.loc[self._curr_page, 'class'] = x
            self.next()

    def save(self):
        self.df.to_csv(self._dir + '/' + 'labels.csv', header=None, index=None)
        messagebox.showinfo("Info", "File saved as {}".format(self._dir + '/labels.csv'))

    def chg_image(self):
        if self.im.mode == "1":
            self.img = ImageTk.BitmapImage(self.im, foreground="white")
        else:
            self.img = ImageTk.PhotoImage(self.im)
        self.la.config(image=self.img, bg="#000000",
            width=self.img.width(), height=self.img.height())

    def open(self):
        dir = filedialog.askdirectory()

        if dir is not None:
            self._dir = dir
            files = glob.glob(dir + '/*.jpg')

            self.num_pages = len(files)

            self.df = pd.DataFrame(files, columns=['filename'])
            self.df['class'] = files

            if self.num_pages:
                self._curr_page = 0
                self.im = Image.open(self.df.loc[self._curr_page, 'filename'])

                self.chg_image()

    def prev(self):
        if self._curr_page > 0:
            self._curr_page -= 1

            self.im = Image.open(self.df.loc[self._curr_page, 'filename'])
            self.chg_image()

    def next(self):
        if self._curr_page < self.num_pages - 1:
            self._curr_page += 1

            self.im = Image.open(self.df.loc[self._curr_page, 'filename'])
            self.chg_image()


if __name__ == "__main__":
    app = App()
    app.mainloop()

