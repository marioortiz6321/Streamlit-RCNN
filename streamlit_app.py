# Importamos librerias
import cv2
import numpy as np
import streamlit as st

# Cargamos el modelo COCO 80 clases
rcnn = cv2.dnn.readNetFromTensorflow('DNN/frozen_inference_graph.pb',
                                     'DNN/mask_rcnn_inception_v2_coco_2018_01_28.pbtxt')

# Cargar archivo de nombres de clases
with open('DNN/coco_labels.txt', 'r') as f:
    classes = f.read().strip().split('\n')

# Widget de carga de archivos
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "png"])
if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    alto, ancho, _ = img.shape

    # Diccionario de precios por clase
    precios = {
        'pizza': 20000.00,
        'hot dog': 10000.00,
        'donut': 4500.00,
        'hamburguesa': 7500.00,
        'cake': 9000.00,
        'bowl': 9000.00,
        'bottle': 3500.00,
        'wine glass': 19500.00,
        'apple': 2500.00
        # Añadir más clases y precios según sea necesario
    }

    # Generamos los colores
    colores = np.random.randint(0, 255, (90, 3))

    # Alistamos nuestra imagen
    blob = cv2.dnn.blobFromImage(img, swapRB=True)  # Swap: BGR -> RGB

    # Procesamos la imagen
    rcnn.setInput(blob)

    # Extraemos los Rect y Mascaras
    info, masks = rcnn.forward(["detection_out_final", "detection_masks"])

    # Extraemos la cantidad de objetos detectados
    contObject = info.shape[2]

    # Lista para almacenar los alimentos detectados y sus precios
    alimentos_detectados = []

    # Iteramos sobre los objetos detectados
    for i in range(contObject):
        # Extraemos los rectangulos de los objetos
        inf = info[0, 0, i]

        # Extraemos Clase
        clase = int(inf[1])
        label = classes[clase]

        # Verificamos si la clase tiene un precio asociado
        if label in precios:
            precio_objeto = precios[label]

        else:
            precio_objeto = 0.00

        # Extraemos puntaje
        puntaje = inf[2]



        # Filtro
        if puntaje < 0.7:
            continue

        # Coordenadas del Rectangulos para deteccion de objetos
        x = int(inf[3] * ancho)
        y = int(inf[4] * alto)
        x2 = int(inf[5] * ancho)
        y2 = int(inf[6] * alto)

        # Extraemos el tamaño de los objetos
        tamobj = img[y:y2, x:x2]
        tamalto, tamancho, _ = tamobj.shape

        # Extraemos Mascara
        mask = masks[i, clase]
        mask = cv2.resize(mask, (tamancho, tamalto))

        # Establecemos un umbral
        _, mask = cv2.threshold(mask, 0.15, 255, cv2.THRESH_BINARY)
        mask = np.array(mask, np.uint8)

        # Extraemos coordenadas de la mascara
        contornos, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Elegimos los colores
        color = colores[clase]
        r = int(color[0])
        g = int(color[1])
        b = int(color[2])

        # Iteramos los contornos
        for cont in contornos:
            # Dibujamos mascara con transparencia
            overlay = tamobj.copy()
            cv2.fillPoly(overlay, [cont], (r, g, b), lineType=cv2.LINE_AA)

            # Aplicamos la máscara con transparencia a la región de interés en la imagen original
            cv2.addWeighted(overlay, 0.6, tamobj, 0.6, 0, tamobj)

            # Dibujamos
            cv2.rectangle(img, (x, y), (x2, y2), (r, g, b), 2)

            # Calculamos el tamaño del texto
            (text_width, text_height), _ = cv2.getTextSize(f'{label}: {puntaje:.2f}: ${precio_objeto:.2f}',
                                                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

            # Dibujamos un rectángulo lleno detrás del texto
            cv2.rectangle(img, (x, y - text_height - 10), (x + text_width, y), (0, 0, 0), -1)

            # Añadimos el texto con el label, puntaje y precio
            cv2.putText(img, f'{label}: {puntaje:.2f}: ${precio_objeto:.2f}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 255), 1)

            # Añadimos el alimento y su precio a la lista
            alimentos_detectados.append((label, precio_objeto))
    # Convertimos la imagen de BGR a RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Mostramos la imagen en Streamlit
    st.image(img, use_column_width=True)

    # Calculamos el precio total
    precio_total = sum(precio for _, precio in alimentos_detectados)

    # Mostramos el recibo en Streamlit
    st.markdown("<h2 style='text-align: center;'>Receipt</h2>", unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)
    for alimento, precio in alimentos_detectados:
        st.markdown(f"<p style='text-align: center;'>{alimento}: ${precio:.2f}</p>", unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(f"<p style='text-align: center;'><b>Total: ${precio_total:.2f}</b></p>", unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)

    # Creamos un archivo de texto con el recibo
    with open('Receipt.txt', 'w') as f:
        f.write("Receipt\n")
        for alimento, precio in alimentos_detectados:
            f.write(f"{alimento}: ${precio:.2f}\n")
        f.write(f"Total: ${precio_total:.2f}\n")

    # Proporcionamos un botón de descarga para el recibo
    with open('Receipt.txt', 'r') as f:
        recibo_texto = f.read()
    st.download_button(
        label="Download Receipt",
        data=recibo_texto,
        file_name='Receipt.txt',
        mime='text/plain',
    )