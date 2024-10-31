from tkinter import *
import tkinter.filedialog
from PIL import Image, ImageFilter, ImageTk
import os
import tkinter.messagebox
import tkinter.ttk
import numpy as np
import cv2 as cv

sizex = 0
sizey = 0
quality = 100
path = ''
output_path = None
output_file = None
root = Tk()
root.geometry()
label_img = None

root.title('图片智能修复')


class Sketcher:

    def __init__(self, windowname, dests, colors_func):
        self.prev_pt = None
        self.windowname = windowname
        self.dests = dests
        self.colors_func = colors_func
        self.dirty = False
        self.show()
        cv.setMouseCallback(self.windowname, self.on_mouse)

    def show(self):
        cv.imshow(self.windowname, self.dests[0])
        # cv.imshow(self.windowname + ": mask", self.dests[1])


    def on_mouse(self, event, x, y, flags, param):
        pt = (x, y)
        if event == cv.EVENT_LBUTTONDOWN:
            self.prev_pt = pt
        elif event == cv.EVENT_LBUTTONUP:
            self.prev_pt = None

        if self.prev_pt and flags & cv.EVENT_FLAG_LBUTTON:
            for dst, color in zip(self.dests, self.colors_func()):
                cv.line(dst, self.prev_pt, pt, color, 5)
            self.dirty = True
            self.prev_pt = pt
            self.show()



def loadimg():
    global path
    global sizex
    global sizey
    path = tkinter.filedialog.askopenfilename()
    lb.config(text=path)
    if path != '':
        try:
            img = Image.open(path)
            sizex = img.size[0]
            sizey = img.size[1]
            img = img.resize((400, 400), Image.ANTIALIAS)
            global img_origin
            img_origin = ImageTk.PhotoImage(img)
            global label_img
            label_img.configure(image=img_origin)
            label_img.pack()

        except OSError:
            tkinter.messagebox.showerror('错误', '图片格式错误，无法识别')


def inpaint(path):
    def function(img):
        try:

            # 创建一个原始图像的副本
            img_mask = img.copy()
            # 创建原始图像的黑色副本
            # Acts as a mask
            inpaintMask = np.zeros(img.shape[:2], np.uint8)

            # Create sketch using OpenCV Utility Class: Sketcher
            sketch = Sketcher('image', [img_mask, inpaintMask], lambda: ((255, 255, 255), 255))

            ch = cv.waitKey()

            if ch == ord('t'):

                res = cv.inpaint(src=img_mask, inpaintMask=inpaintMask, inpaintRadius=3, flags=cv.INPAINT_TELEA)
                cv.imshow('Inpaint Output using FMM', res)
                cv.waitKey()
                cv.imwrite(path, res)

            if ch == ord('n'):

                res = cv.inpaint(src=img_mask, inpaintMask=inpaintMask, inpaintRadius=3, flags=cv.INPAINT_NS)
                cv.imshow('Inpaint Output using NS Technique', res)
                cv.waitKey()
                cv.imwrite(path, res)

            if ch == ord('r'):
                img_mask[:] = img
                inpaintMask[:] = 0
                sketch.show()

            cv.destroyAllWindows()
        except ValueError as e:
            tkinter.messagebox.showerror('', repr(e))

    if path != '':
        try:
            img = Image.open(path)
            img1 = cv.imread(path, cv.IMREAD_COLOR)
            img1 = function(img1)


        except OSError:
            lb.config(text="您没有选择任何文件")
            tkinter.messagebox.showerror('错误', '图片格式错误，无法识别')

    else:
        tkinter.messagebox.showerror('错误', '未发现路径')


lb = Label(root, text='会在原路径保存图像')
lb.pack()

lb1 = Label(root, text='警告：会覆盖原图片', width=27, height=2, font=("Arial", 10), bg="red")
lb1.pack(side='top')

btn = Button(root, text="选择图片", command=loadimg)
btn.pack()

lb2 = Label(root, text='按下开始绘制想要修复的位置')
lb2.pack()

btn2 = Button(root, text="开始", command=lambda: inpaint(path))
btn2.pack()

lb3 = Label(root, text='绘制完成使用以下步骤')
lb3.pack()

lb4 = Label(root, text='t-使用FMM修复\nn-使用NS方法修复\nr-重新绘制区域')
lb4.pack()

label_img = tkinter.Label(root, text='原始图片')
label_img.pack()

root.mainloop()


