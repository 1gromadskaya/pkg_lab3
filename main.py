import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from tkinter import PhotoImage
from PIL import Image, ImageTk
import os
import cv2
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


class ImageProcessorApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Обработка изображений")
        self.master.geometry("1300x900")
        self.image_folder = ""
        self.image_list = []
        self.current_image_index = 0
        self.original_image = None
        self.base_image = None
        self.processed_image = None
        self.alpha = 1.0
        self.beta = 0
        self.filter_type = "Медианный"
        self.kernel_size = 3
        self.equalize_method = tk.StringVar()
        self.equalize_method.set("RGB")
        self.filters = []
        self.create_widgets()

    def create_widgets(self):
        self.notebook = ttk.Notebook(self.master)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        self.image_tab = tk.Frame(self.notebook)
        self.notebook.add(self.image_tab, text="Изображения")
        self.top_frame = tk.Frame(self.image_tab)
        self.top_frame.pack(side=tk.TOP, fill=tk.X, pady=5)
        self.folder_btn = self.create_button(self.top_frame, "Выбрать папку", self.load_folder)
        self.folder_btn.pack(side=tk.LEFT, padx=5)
        self.prev_btn = self.create_button(self.top_frame, "Предыдущее", self.show_prev_image)
        self.prev_btn.pack(side=tk.LEFT, padx=5)
        self.next_btn = self.create_button(self.top_frame, "Следующее", self.show_next_image)
        self.next_btn.pack(side=tk.LEFT, padx=5)
        self.display_frame = tk.Frame(self.image_tab)
        self.display_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.original_canvas = tk.Canvas(self.display_frame, width=500, height=500, bg='gray')
        self.original_canvas.pack(side=tk.LEFT, padx=10, pady=10)
        self.processed_canvas = tk.Canvas(self.display_frame, width=500, height=500, bg='gray')
        self.processed_canvas.pack(side=tk.RIGHT, padx=10, pady=10)
        self.hist_frame = tk.Frame(self.image_tab)
        self.hist_frame.pack(side=tk.TOP, fill=tk.X, pady=5)
        self.original_hist_canvas = FigureCanvasTkAgg(Figure(figsize=(5, 2)), master=self.hist_frame)
        self.original_hist_canvas.get_tk_widget().pack(side=tk.LEFT, padx=10)
        self.processed_hist_canvas = FigureCanvasTkAgg(Figure(figsize=(5, 2)), master=self.hist_frame)
        self.processed_hist_canvas.get_tk_widget().pack(side=tk.RIGHT, padx=10)
        self.controls_frame = tk.Frame(self.master)
        self.controls_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=10)
        contrast_frame = tk.LabelFrame(self.controls_frame, text="Линейное контрастирование", padx=10, pady=10)
        contrast_frame.grid(row=0, column=0, padx=10, pady=5, sticky="w")
        tk.Label(contrast_frame, text="Контраст (alpha)").grid(row=0, column=0, padx=5, pady=5)
        self.alpha_slider = tk.Scale(contrast_frame, from_=0.1, to=3.0, resolution=0.1, orient=tk.HORIZONTAL, command=self.update_alpha)
        self.alpha_slider.set(1.0)
        self.alpha_slider.grid(row=0, column=1, padx=5, pady=5)
        tk.Label(contrast_frame, text="Яркость (beta)").grid(row=1, column=0, padx=5, pady=5)
        self.beta_slider = tk.Scale(contrast_frame, from_=-100, to=100, resolution=1, orient=tk.HORIZONTAL, command=self.update_beta)
        self.beta_slider.set(0)
        self.beta_slider.grid(row=1, column=1, padx=5, pady=5)
        self.apply_contrast_btn = tk.Button(contrast_frame, text="Применить контрастирование", command=self.apply_linear_contrast)
        self.apply_contrast_btn.grid(row=2, column=0, columnspan=2, pady=5)
        hist_eq_frame = tk.LabelFrame(self.controls_frame, text="Выравнивание гистограммы", padx=10, pady=10)
        hist_eq_frame.grid(row=0, column=1, padx=10, pady=5, sticky="w")
        self.rgb_radio = tk.Radiobutton(hist_eq_frame, text="RGB пространство", variable=self.equalize_method, value="RGB")
        self.rgb_radio.grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.hsv_radio = tk.Radiobutton(hist_eq_frame, text="HSV пространство", variable=self.equalize_method, value="HSV")
        self.hsv_radio.grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.apply_hist_eq_btn = tk.Button(hist_eq_frame, text="Применить эквализацию гистограммы", command=self.apply_histogram_equalization)
        self.apply_hist_eq_btn.grid(row=2, column=0, padx=5, pady=5)
        filter_frame = tk.LabelFrame(self.controls_frame, text="Нелинейные фильтры", padx=10, pady=10)
        filter_frame.grid(row=0, column=2, padx=10, pady=5, sticky="w")
        tk.Label(filter_frame, text="Тип фильтра").grid(row=0, column=0, padx=5, pady=5)
        self.filter_combo = ttk.Combobox(filter_frame, values=["Медианный", "Минимальный", "Максимальный"])
        self.filter_combo.current(0)
        self.filter_combo.grid(row=0, column=1, padx=5, pady=5)
        self.filter_combo.bind("<<ComboboxSelected>>", self.update_filter_type)
        tk.Label(filter_frame, text="Размер ядра").grid(row=1, column=0, padx=5, pady=5)
        self.kernel_slider = tk.Scale(filter_frame, from_=3, to=15, resolution=2, orient=tk.HORIZONTAL, command=self.update_kernel_size)
        self.kernel_slider.set(3)
        self.kernel_slider.grid(row=1, column=1, padx=5, pady=5)
        self.apply_filter_btn = tk.Button(filter_frame, text="Применить фильтр", command=self.apply_non_linear_filter)
        self.apply_filter_btn.grid(row=2, column=0, columnspan=2, pady=5)
        save_reset_frame = tk.Frame(self.controls_frame)
        save_reset_frame.grid(row=0, column=3, padx=10, pady=5, sticky="w")
        self.save_btn = self.create_button(save_reset_frame, "Сохранить", self.save_image)
        self.save_btn.pack(side=tk.TOP, padx=5, pady=5)
        self.reset_btn = self.create_button(save_reset_frame, "Сбросить", self.reset_image)
        self.reset_btn.pack(side=tk.TOP, padx=5, pady=5)

    def create_button(self, parent, text, command):
        return tk.Button(parent, text=text, command=command, width=20, height=2, relief="solid", bd=2)

    def load_folder(self):
        self.image_folder = filedialog.askdirectory()
        if self.image_folder:
            supported_formats = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
            self.image_list = [f for f in os.listdir(self.image_folder) if f.lower().endswith(supported_formats)]
            self.image_list.sort()
            if self.image_list:
                self.current_image_index = 0
                self.filters = []
                self.reset_contrast_brightness()
                self.load_image()
            else:
                messagebox.showerror("Ошибка", "В папке нет изображений поддерживаемых форматов.")

    def load_image(self):
        if not self.image_list:
            return
        image_path = os.path.join(self.image_folder, self.image_list[self.current_image_index])
        if not os.path.exists(image_path):
            messagebox.showerror("Ошибка", f"Файл не найден: {image_path}")
            return
        self.original_image = cv2.imread(image_path)
        if self.original_image is None:
            messagebox.showerror("Ошибка", f"Не удалось загрузить изображение: {image_path}")
            return
        self.reset_modifications()
        self.base_image = self.original_image.copy()
        self.processed_image = self.original_image.copy()
        self.display_image_on_canvas(self.original_image, self.original_canvas)
        self.display_image_on_canvas(self.processed_image, self.processed_canvas)
        self.update_histogram(self.original_image, self.original_hist_canvas)
        self.update_histogram(self.processed_image, self.processed_hist_canvas)

    def reset_contrast_brightness(self):
        self.alpha_slider.set(1.0)
        self.beta_slider.set(0)
        self.update_alpha()
        self.update_beta()

    def display_image_on_canvas(self, image, canvas):
        if image is None:
            return
        h, w = image.shape[:2]
        canvas_width = canvas.winfo_width()
        canvas_height = canvas.winfo_height()
        if canvas_width == 1 or canvas_height == 1:
            canvas_width = 500
            canvas_height = 500
        ratio = min(canvas_width / w, canvas_height / h)
        new_width = int(w * ratio)
        new_height = int(h * ratio)
        resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        image_rgb = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image_rgb)
        image_tk = ImageTk.PhotoImage(image_pil)
        canvas.delete("all")
        canvas.create_image(canvas_width // 2, canvas_height // 2, anchor=tk.CENTER, image=image_tk)
        canvas.image_tk = image_tk

    def update_alpha(self, value=None):
        self.alpha = float(value) if value else self.alpha
        self.apply_linear_contrast()

    def update_beta(self, value=None):
        self.beta = int(value) if value else self.beta
        self.apply_linear_contrast()

    def apply_linear_contrast(self):
        if self.original_image is None:
            return
        self.processed_image = cv2.convertScaleAbs(self.base_image, alpha=self.alpha, beta=self.beta)
        self.display_image_on_canvas(self.processed_image, self.processed_canvas)
        self.update_histogram(self.processed_image, self.processed_hist_canvas)

    def apply_histogram_equalization(self):
        if self.equalize_method.get() == "RGB":
            self.processed_image = self.histogram_equalization_rgb(self.base_image)
        elif self.equalize_method.get() == "HSV":
            self.processed_image = self.histogram_equalization_hsv(self.base_image)
        self.display_image_on_canvas(self.processed_image, self.processed_canvas)
        self.update_histogram(self.processed_image, self.processed_hist_canvas)

    def histogram_equalization_rgb(self, image):
        img_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
        img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
        return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

    def histogram_equalization_hsv(self, image):
        img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        img_hsv[:, :, 2] = cv2.equalizeHist(img_hsv[:, :, 2])
        return cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)

    def update_histogram(self, image, hist_canvas):
        if image is None:
            return
        hist_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([hist_image], [0], None, [256], [0, 256])
        fig = hist_canvas.figure
        fig.clf()
        ax = fig.add_subplot(111)
        ax.clear()
        ax.plot(hist)
        ax.set_xlim([0, 256])
        hist_canvas.draw()

    def apply_non_linear_filter(self):
        if self.processed_image is None:
            return
        kernel = np.ones((self.kernel_size, self.kernel_size), np.uint8)
        if self.filter_type == "Медианный":
            self.processed_image = cv2.medianBlur(self.base_image, self.kernel_size)
        elif self.filter_type == "Минимальный":
            self.processed_image = cv2.erode(self.base_image, kernel)
        elif self.filter_type == "Максимальный":
            self.processed_image = cv2.dilate(self.base_image, kernel)
        self.display_image_on_canvas(self.processed_image, self.processed_canvas)
        self.update_histogram(self.processed_image, self.processed_hist_canvas)

    def update_filter_type(self, event):
        self.filter_type = self.filter_combo.get()

    def update_kernel_size(self, value):
        self.kernel_size = int(value)
        self.apply_non_linear_filter()

    def reset_image(self):
        if self.original_image is None:
            return
        self.processed_image = self.original_image.copy()
        self.display_image_on_canvas(self.processed_image, self.processed_canvas)
        self.update_histogram(self.processed_image, self.processed_hist_canvas)

    def show_prev_image(self):
        if not self.image_list:
            return
        self.current_image_index = (self.current_image_index - 1) % len(self.image_list)
        self.load_image()

    def show_next_image(self):
        if not self.image_list:
            return
        self.current_image_index = (self.current_image_index + 1) % len(self.image_list)
        self.load_image()

    def save_image(self):
        if self.processed_image is None:
            return
        file_path = filedialog.asksaveasfilename(defaultextension=".png",
                                                 filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg"),
                                                            ("All files", "*.*")])
        if file_path:
            cv2.imwrite(file_path, self.processed_image)


if __name__ == "__main__":
    root = tk.Tk()
    app = ImageProcessorApp(root)
    root.mainloop()
