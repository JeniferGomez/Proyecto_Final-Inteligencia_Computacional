import tkinter as tk
from pathlib import Path
from tkinter import Entry, Button, PhotoImage, messagebox, filedialog
try:
    from PIL import ImageTk, Image, ImageGrab
except ImportError:
    print("El módulo Pillow (PIL) no está disponible.")
try:
    import numpy as np
except ImportError:
    print("El módulo Pillow (PIL) no está disponible.")
from keras.models import load_model
import json
import torch
import torch.nn as nn
from torchvision import models, transforms
import numpy as np
import os

# Cargar una red preentrenada (en este caso, ResNet-50)
model = models.resnet50(pretrained=True)
model.eval()  # Establecer en modo de evaluación

selected_image = None

# Define una red siamesa
class SiameseNetwork(nn.Module):
    def __init__(self, pretrained_model):
        super(SiameseNetwork, self).__init__()
        self.model = pretrained_model

    def forward_one(self, x):
        return self.model(x)

    def forward(self, input1, input2):
        output1 = self.forward_one(input1)
        output2 = self.forward_one(input2)
        return output1, output2

# Función para extraer características de una imagen
def extract_features(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Redimensionar todas las imágenes a 224x224
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)  # Agregar una dimensión de lote
    features = model(image)
    return features.detach().numpy().flatten()

# Directorio de datos
data_dir = "assets_res/SEÑALES DE TRANSITO/REGLAMENTARIAS/"  # Cambia esto a la ubicación de tu conjunto de datos

# Lista de imágenes y características
image_paths = []
image_features = []

# Cargar imágenes y extraer características
for root, _, files in os.walk(data_dir):
    for file in files:
        if file.endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(root, file)
            image_paths.append(image_path)
            features = extract_features(image_path)
            image_features.append(features)

# Convertir a matrices numpy
image_features = np.array(image_features)

# Crear una red siamesa
siamese_net = SiameseNetwork(model)

# Guardar el modelo entrenado en un archivo .pth
torch.save(siamese_net.state_dict(), 'siamese_model.pth')

left_canvas = None
right_canvas = None

def classify_image(file_path):
    try:
        # Cargar el modelo previamente entrenado
        model = load_model("./assets_cla/traffic_classifier14.h5")

        # Cargar el diccionario de clases desde el archivo JSON
        def cargar_datos_desde_json(ruta_json):
            try:
                with open(ruta_json, 'r', encoding='utf-8') as archivo_json:
                    datos = json.load(archivo_json)
                return datos
            except FileNotFoundError:
                return None

        classes = cargar_datos_desde_json("./assets_cla/clases.json")

        image = Image.open(file_path)
        image = image.resize((30, 30))

        if image.mode != "RGB":
            image = image.convert("RGB")

        image = np.array(image) / 255.0

        if image.shape[-1] != 3:
            raise ValueError("Input image should have 3 color channels (RGB)")

        image = np.expand_dims(image, axis=0)
        pred = model.predict(image)
        pred_class = np.argmax(pred, axis=1)[0] + 1
        sign_data = classes.get(str(pred_class))

        return pred_class, sign_data
    except Exception as e:
        print(f"Error during classification: {e}")
        return None, None
def toggle_elements():
    global left_canvas, right_canvas, root

    # Elimina los elementos existentes
    #left_canvas.delete("all")
    #right_canvas.delete("all")

    right_canvas.destroy()
    # Modifica el tamaño de la ventana principal
    root.geometry("1012x606")

    # Creamos un Canvas para el lado derecho de la ventana (color azul)
    right_canvas = tk.Canvas(root, bg="#FFFFFF", width=662, height=606)
    right_canvas.pack(side="left", fill="both", expand=True)
    # Modifica los tamaños de los canvas
    left_canvas.config(width=350, height=606)

    right_canvas.create_text(
        20.0,
        20.0,
        anchor="nw",
        text="🔍 Clasifica Señales de Transito.",
        fill="#253B66",
        font=("Montserrat Bold", 30 * -1),
    )

    right_canvas.create_text(
        20.0,
        70.0,
        anchor="nw",
        text="Te invitamos a cargar una imagen de una señal de tráfico en nuestra aplicación, y nuestro ",
        fill="#121D33",
        font=("Montserrat Bold", 16 * -1),
    )

    right_canvas.create_text(
        20.0,
        90.0,
        anchor="nw",
        text="avanzado modelo se encargará de identificarla y proporcionarte su nombre junto con una",
        fill="#121D33",
        font=("Montserrat Bold", 16 * -1),
    )

    right_canvas.create_text(
        20.0,
        110.0,
        anchor="nw",
        text="breve descripción para una comprensión más completa. 😉",
        fill="#121D33",
        font=("Montserrat Bold", 16 * -1),
    )

    # Agregar el botón en el lado izquierdo para volver
    button_image_3 = tk.PhotoImage(file="assets_cla/subir.png")
    right_canvas.button_image_2 = button_image_3
    button_3 = tk.Button(
        right_canvas,
        image=button_image_3,
        borderwidth=0,
        highlightthickness=0,
        relief="flat",
        command=load_image
    )
    button_3.place(x=180.0, y=150.0, width=198.0, height=56.0)

    right_canvas.loaded_image = None  # Variable para almacenar la imagen cargada
    right_canvas.image_label = None  # Etiqueta para mostrar la imagen cargada

    # Etiqueta para mostrar el nombre de la señal
    right_canvas.nombre_label = tk.Label(
        right_canvas,
        anchor="nw",
        text="",
        fg="#000000",  # Cambia -fill a -fg para establecer el color del texto
        font=("Montserrat Bold", 16 * -1),
    )
    right_canvas.nombre_label.place(x=140.0, y=420.0)

    # Etiqueta para mostrar la descripción de la señal
    right_canvas.descripcion_label = tk.Label(
        right_canvas,
        anchor="nw",
        text="Descripción:",
        fg="#000000",  # Cambia -fill a -fg para establecer el color del texto
        font=("Montserrat Bold", 16 * -1),
        wraplength=600,  # Ajusta este valor según tus preferencias
    )
    right_canvas.descripcion_label.place(x=20.0, y=450.0)

def load_image():
    global right_canvas  # Accede a la variable global right_canvas
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png *.jpg *.jpeg *.gif *.bmp")])
    if file_path:
        # Abre la imagen con Pillow (PIL)
        pil_image = Image.open(file_path)

        # Redimensiona la imagen al tamaño deseado sin antialiasing
        resized_image = pil_image.resize((250, 180), Image.NEAREST)

        # Convierte la imagen de Pillow a PhotoImage de Tkinter
        image = ImageTk.PhotoImage(resized_image)

        right_canvas.loaded_image = image
        if right_canvas.image_label:
            right_canvas.delete(right_canvas.image_label)  # Elimina la imagen anterior
        right_canvas.image_label = right_canvas.create_image(300.0, 300.0, image=image)

        # Realiza la clasificación de la señal y obtiene el nombre y la descripción
        if right_canvas.loaded_image:
            pred_class, sign_data = classify_image(file_path)
            if sign_data:
                nombre = sign_data.get('nombre', '')
                descripcion = sign_data.get('descripcion', '')
                right_canvas.nombre_label.config(text=f" {nombre}")
                right_canvas.descripcion_label.config(text=f"Descripción: {descripcion}")


def toggle_elements2():
    global left_canvas, right_canvas, root

    right_canvas.destroy()
    # Modifica el tamaño de la ventana principal
    root.geometry("1030x606")

    # Creamos un Canvas para el lado derecho de la ventana (color azul)
    right_canvas = tk.Canvas(root, bg="#FFFFFF", width=662, height=606)
    right_canvas.pack(side="left", fill="both", expand=True)
    # Modifica los tamaños de los canvas
    left_canvas.config(width=350, height=606)

    right_canvas.create_text(
        20.0,
        20.0,
        anchor="nw",
        text="🛠️ Restaura Señales de Tránsito.",
        fill="#253B66",
        font=("Montserrat Bold", 30 * -1),
    )

    right_canvas.create_text(
        20.0,
        70.0,
        anchor="nw",
        text="Hemos integrado avanzados modelos de inteligencia artificial generativa en nuestra ",
        fill="#121D33",
        font=("Montserrat Bold", 16 * -1),
    )

    right_canvas.create_text(
        20.0,
        90.0,
        anchor="nw",
        text="aplicación para restaurar señales de tráfico deterioradas. Simplemente captura una foto de",
        fill="#121D33",
        font=("Montserrat Bold", 16 * -1),
    )

    right_canvas.create_text(
        20.0,
        110.0,
        anchor="nw",
        text="una señal en mal estado, súbela y observa la magia en acción mientras nuestra app la ",
        fill="#121D33",
        font=("Montserrat Bold", 16 * -1),
        )

    right_canvas.create_text(
        20.0,
        130.0,
        anchor="nw",
        text="devuelve a su estado original. 😉",
        fill="#121D33",
        font=("Montserrat Bold", 16 * -1),
    )

    # Crear un Canvas para mostrar la imagen importada
    right_canvas.image_canvas = tk.Canvas(
        right_canvas,
        bg="#FFFFFF",
        width=300,  # Ancho del Canvas para la imagen
        height=300,  # Alto del Canvas para la imagen
    )
    right_canvas.image_canvas.place(x=20.0, y=220.0)  # Ajusta la posición de la imagen aquí

    right_canvas.image_canvas2 = tk.Canvas(
        right_canvas,
        bg="#FFFFFF",
        width=300,  # Ancho del Canvas para la imagen
        height=300,  # Alto del Canvas para la imagen
    )
    right_canvas.image_canvas2.place(x=350.0, y=220.0)  # Ajusta la posición de la imagen aquí

    def find_similar_images(query_image_path, top_k=1, canvas=None):
        query_features = extract_features(query_image_path)
        if len(image_features) == 0:
            print("No hay imágenes disponibles en el conjunto de datos.")
            return

        if canvas is not None:
            # Limpiar el lienzo antes de mostrar una nueva imagen
            canvas.delete("all")

            # Obtener el ancho y alto del canvas
            canvas_width = canvas.winfo_width()
            canvas_height = canvas.winfo_height()

            # Calcular el aspect ratio de la imagen de consulta
            image = Image.open(query_image_path)
            aspect_ratio = image.width / image.height

            # Fijar el ancho deseado para el canvas
            new_width = 300  # Cambia este valor según tus preferencias

            # Calcular el nuevo alto basado en el aspect ratio
            new_height = int(new_width / aspect_ratio)

            # Redimensionar la imagen para ajustarla al tamaño deseado
            image = image.resize((new_width, new_height), Image.BILINEAR)

            # Mostrar la imagen en el lienzo proporcionado
            similar_image = ImageTk.PhotoImage(image)
            # canvas.create_image(0, 0, anchor=tk.NW, image=similar_image)
            canvas.image = similar_image  # Mantener una referencia para evitar que la imagen sea destruida

            # Calcular la distancia coseno entre la imagen de consulta y todas las imágenes en el conjunto de datos
            distances = np.dot(image_features, query_features) / (
                    np.linalg.norm(image_features, axis=1) * np.linalg.norm(query_features))
            # Obtener los índices de las imágenes más similares
            similar_indices = np.argsort(distances)[::-1][:top_k]

            # Mostrar las imágenes más similares en el lienzo
            for i, idx in enumerate(similar_indices):
                similar_image_path = image_paths[idx]
                similar_image = Image.open(similar_image_path)
                similar_image.thumbnail((new_width, new_height))  # Redimensionar la imagen
                similar_image = ImageTk.PhotoImage(similar_image)

                # Mostrar la imagen similar en el lienzo proporcionado
                # canvas.create_image(0, (i + 1) * (new_height + 1), anchor=tk.NW, image=similar_image)
                canvas.create_image(0, 0, anchor=tk.NW, image=similar_image)
                canvas.image = similar_image  # Mantener una referencia para evitar que la imagen sea destruida

    # Función para procesar la imagen cargada
    def process_image(image_path):
        # Tu código para procesar la imagen aquí
        # Por ejemplo, puedes usar la función find_similar_images
        find_similar_images(image_path, canvas=right_canvas.image_canvas2)  # Llamar a find_similar_images con el lienzo canvas2

    def import_image():
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        if file_path:
            # Cargar la imagen
            image = Image.open(file_path)

            # Obtener las dimensiones del Canvas para la imagen
            canvas_width = right_canvas.image_canvas.winfo_width()
            canvas_height = right_canvas.image_canvas.winfo_height()

            # Redimensionar la imagen para ajustarla al tamaño del Canvas utilizando BILINEAR
            image = image.resize((canvas_width, canvas_height), Image.BILINEAR)

            # Convertir la imagen redimensionada en PhotoImage
            photo = ImageTk.PhotoImage(image)

            # Mostrar la imagen en el Canvas
            right_canvas.image_canvas.create_image(0, 0, anchor="nw", image=photo)
            right_canvas.image_canvas.photo = photo  # Mantener una referencia a la imagen

            # messagebox.showinfo("Imagen Importada", "La imagen se ha importado con éxito.")

            # Procesar la imagen cargada y mostrar el resultado en image_canvas2
            process_image(file_path)

    button_image_1 = tk.PhotoImage(file="assets_res/subir.png")
    right_canvas.button1_image = button_image_1  # Mantén una referencia a la imagen del botón 1
    button_1 = Button(
        right_canvas,
        image=button_image_1,
        borderwidth=0,
        highlightthickness=0,
        relief="flat",
        command=import_image  # Asigna la función de importación al botón
    )
    button_1.place(x=200.0, y=160.0, width=105.0, height=48.0)

    button_image_11 = tk.PhotoImage(file="assets_gui/Clasif.png")
    right_canvas.button11_image = button_image_11  # Mantén una referencia a la imagen del botón 1
    button_11 = Button(
        right_canvas,
        image=button_image_11,
        borderwidth=0,
        highlightthickness=0,
        relief="flat",
        command=download_image_from_canvas  # Asigna la función de importación al botón
    )
    button_11.place(x=200.0, y=550.0, width=115.0, height=48.0)

def download_image_from_canvas():
    if right_canvas.image_canvas2.winfo_exists():  # Verifica si el lienzo existe
        # Define las coordenadas manualmente (ajusta estos valores según tus necesidades)
        x = 1290  # Coordenada X del lienzo en la ventana
        y = 420  # Coordenada Y del lienzo en la ventana
        width = 300  # Ancho del lienzo
        height = 300  # Alto del lienzo

        # Captura la pantalla dentro del área del lienzo
        screenshot = ImageGrab.grab(bbox=(x, y, x + width, y + height))

        # Guarda la imagen en un archivo
        file_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png")])
        if file_path:
            screenshot.save(file_path)
            messagebox.showinfo("Descarga Exitosa", "La imagen se ha descargado con éxito.")
        else:
            messagebox.showinfo("Descarga Cancelada", "La descarga de la imagen ha sido cancelada.")
    else:
        messagebox.showinfo("Sin Imagen", "No hay ninguna imagen para descargar.")



def create_divided_window():
    global left_canvas, right_canvas, root

    root = tk.Tk()
    root.geometry("700x506")  # Tamaño de la ventana
    root.title("App GoMog")

    # Creamos un Canvas para el lado izquierdo de la ventana (color rojo)
    left_canvas = tk.Canvas(root, bg="#5E95FF", width=350, height=506)
    left_canvas.pack(side="left", fill="both", expand=True)

    # Creamos un Canvas para el lado derecho de la ventana (color azul)
    right_canvas = tk.Canvas(root, bg="#FFFFFF", width=350, height=506)
    right_canvas.pack(side="left", fill="both", expand=True)

    # Cargar y mostrar una imagen en el lado derecho
    image_image_1 = tk.PhotoImage(file="assets_gui/image_1.png")
    right_canvas.image = image_image_1
    image_1 = right_canvas.create_image(180, 240, image=image_image_1)

    # Mostrar el texto en el lado izquierdo
    left_canvas.create_text(
        10.0,
        40.0,
        anchor="nw",
        text="TransitAI",
        fill="#FFFFFF",
        font=("Montserrat Bold", 50 * -1),
    )

    left_canvas.create_text(
        10.0,
        120.0,
        anchor="nw",
        text="Este programa utiliza la inteligencia",
        fill="#FFFFFF",
        font=("Montserrat Regular", 18 * -1),
    )

    left_canvas.create_text(
        10.0,
        149.0,
        anchor="nw",
        text="artificial generativa para restaurar",
        fill="#FFFFFF",
        font=("Montserrat Regular", 18 * -1),
    )

    left_canvas.create_text(
        10.0,
        178.0,
        anchor="nw",
        text="y clasificar imágenes de señales",
        fill="#FFFFFF",
        font=("Montserrat Regular", 18 * -1),
    )

    left_canvas.create_text(
        10.0,
        207.0,
        anchor="nw",
        text="de tránsito de Colombia.",
        fill="#FFFFFF",
        font=("Montserrat Regular", 18 * -1),
    )

    left_canvas.create_text(
        20.0,
        481.0,
        anchor="nw",
        text="© JeniferG & AngieM, 2023",
        fill="#FFFFFF",
        font=("Montserrat Bold", 18 * -1),
    )

    # Agregar el botón en el lado izquierdo para alternar los elementos
    button_image_1 = tk.PhotoImage(file="assets_cla/clasificar.png")
    left_canvas.button_image_1 = button_image_1
    button_1 = tk.Button(
        left_canvas,
        image=button_image_1,
        borderwidth=0,
        highlightthickness=0,
        relief="flat",
        command=toggle_elements
    )
    button_1.place(x=20.0, y=250.0, width=198.0, height=56.0)

    button_image_2 = tk.PhotoImage(file="assets_cla/Restaurar.png")
    left_canvas.button_image_2 = button_image_2
    button_2 = tk.Button(
        left_canvas,
        image=button_image_2,
        borderwidth=0,
        highlightthickness=0,
        relief="flat",
        command=toggle_elements2
    )
    button_2.place(x=20.0, y=310.0, width=198.0, height=56.0)

    def cerrar_ventana():
        root.destroy()

    button_image_22 = tk.PhotoImage(file="assets_cla/salir.png")
    left_canvas.button_image_22 = button_image_22
    button_22 = tk.Button(
        left_canvas,
        image=button_image_22,
        borderwidth=0,
        highlightthickness=0,
        relief="flat",
        command=cerrar_ventana
    )
    button_22.place(x=20.0, y=370.0, width=198.0, height=56.0)


    # Iniciamos el bucle principal de la aplicación
    root.mainloop()

# Llamamos a la función para crear la ventana dividida sin márgenes
create_divided_window()