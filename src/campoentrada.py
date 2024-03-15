import cv2 as cv
import numpy as np
import os

def cargar_imagenes_y_etiquetas():
    # Crear listas para almacenar las imágenes y las etiquetas correspondientes
    imagenes = []
    etiquetas = []

    # Ruta base de la carpeta donde están las imágenes de tus familiares
    ruta_base = "C:/Users/Compumar/Desktop/ReconocimientoFacial/datos/familia/"

    
    
    
    for i in range(1, 10):  # Suponiendo que tienes 5 imágenes de la primera persona
        ruta_imagen = ruta_base + "persona1/WhatsApp Image 2024-03-14 at 8.18.54 PM (1).jpeg"
        if not os.path.exists(ruta_imagen):
            print("La imagen no existe en la ruta:", ruta_imagen)
            continue
        imagen = cv.imread(ruta_imagen)
        if imagen is None:
            print("No se pudo cargar la imagen:", ruta_imagen)
            continue
        # Convertir la imagen a escala de grises
        imagen_gris = cv.cvtColor(imagen, cv.COLOR_BGR2GRAY)
        imagenes.append(imagen_gris)
        etiquetas.append(0)  # Asigna la etiqueta 0 para la primera persona

    
    for i in range(1, 8):  # Suponiendo que tienes 3 imágenes de la segunda persona
        ruta_imagen = ruta_base + "persona2/WhatsApp Image 2024-03-14 at 8.15.39 PM.jpeg"
        if not os.path.exists(ruta_imagen):
            print("La imagen no existe en la ruta:", ruta_imagen)
            continue
        imagen = cv.imread(ruta_imagen)
        if imagen is None:
            print("No se pudo cargar la imagen:", ruta_imagen)
            continue
        # Convertir la imagen a escala de grises
        imagen_gris = cv.cvtColor(imagen, cv.COLOR_BGR2GRAY)
        imagenes.append(imagen_gris)
        etiquetas.append(1)  # Asigna la etiqueta 1 para la segunda persona
    
    
    
    print("Cantidad de imágenes cargadas:", len(imagenes))
    print("Cantidad de etiquetas cargadas:", len(etiquetas))
    
    return imagenes, etiquetas

def entrenar_modelo(imagenes, etiquetas):
    # Crear el modelo
    modelo = cv.face.LBPHFaceRecognizer_create()
    # Entrenar el modelo
    modelo.train(imagenes, np.array(etiquetas))
    return modelo

def main():
    # Cargar imágenes y etiquetas
    imagenes, etiquetas = cargar_imagenes_y_etiquetas()
    
    print("Cantidad de imágenes cargadas:", len(imagenes))
    print("Cantidad de etiquetas cargadas:", len(etiquetas))

    if not imagenes:
        print("No se pudieron cargar las imágenes.")
        return
    
    # Comprobar si ya existe un modelo entrenado
    if os.path.exists('modelos/modelo.xml'):
        # Cargar el modelo entrenado desde el archivo XML
        modelo = cv.face.LBPHFaceRecognizer_create()
        modelo.read('modelos/modelo.xml')
    else:
        # Entrenar el modelo
        modelo = entrenar_modelo(imagenes, etiquetas)
        # Guardar el modelo entrenado en un archivo XML
        modelo.write('modelos/modelo.xml')
    
    # Inicializar la cámara
    camara = cv.VideoCapture(0)

    # Crear el clasificador de Haar para detección de caras
    ruidos = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')

    while True:
        _, captura = camara.read()

        # Convertir a escala de grises
        grises = cv.cvtColor(captura, cv.COLOR_BGR2GRAY)

        # Detectar caras en la imagen
        caras = ruidos.detectMultiScale(grises, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in caras:
            # Predecir la identidad de la cara
            id_, confianza = modelo.predict(grises[y:y+h, x:x+w])

            # Comprobar si la identidad pertenece a tu familia
            if confianza < 60:  
                if id_ == 0:
                    mensaje = '¡Familiar detectado: Hola Maura!'
                elif id_ == 1:
                    mensaje = '¡Familiar detectado: Hola Pablo!'
                else:
                    mensaje = '¡Identidad desconocida!'
                # Mostrar mensaje en la imagen
                cv.putText(captura, mensaje, (x, y-10), cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            else:
                # Mostrar mensaje de identidad desconocida
                cv.putText(captura, '¡Identidad desconocida!', (x, y-10), cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        # Mostrar el frame
        cv.imshow('Reconocimiento Facial', captura)

        # Salir del bucle al presionar 's'
        if cv.waitKey(1) == ord('s'):
            break

    # Liberar la cámara y cerrar todas las ventanas
    camara.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
