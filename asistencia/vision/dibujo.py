"""Utilidades para dibujar resultados de detección sobre los cuadros de video."""

import unicodedata

import cv2

# Colores BGR
VERDE = (94, 197, 34)
AZUL = (235, 160, 59)
ROJO = (68, 68, 239)
AMARILLO = (21, 204, 250)
BLANCO = (255, 255, 255)
NEGRO = (20, 20, 20)

FUENTE = cv2.FONT_HERSHEY_SIMPLEX


def texto_ascii(texto):
    """Las fuentes Hershey de OpenCV no soportan tildes ni ñ: se transliteran."""
    return unicodedata.normalize('NFKD', str(texto)).encode('ascii', 'ignore').decode('ascii')


def color_texto(fondo):
    """Negro sobre fondos claros y blanco sobre fondos oscuros (luminancia del color BGR)."""
    b, g, r = fondo
    return NEGRO if (0.114 * b + 0.587 * g + 0.299 * r) > 140 else BLANCO


def etiqueta(img, texto, x, y, color, escala=0.55, grosor=1):
    """Dibuja texto con fondo sólido para que sea legible sobre cualquier imagen."""
    texto = texto_ascii(texto)
    (tw, th), base = cv2.getTextSize(texto, FUENTE, escala, grosor)
    y = max(th + base + 2, y)
    cv2.rectangle(img, (x, y - th - base - 4), (x + tw + 8, y + 2), color, cv2.FILLED)
    cv2.putText(img, texto, (x + 4, y - base), FUENTE, escala, color_texto(color), grosor, cv2.LINE_AA)


def esquinas(img, rect, color, grosor=2):
    """Marco tipo visor (solo esquinas) alrededor del rostro."""
    x, y, w, h = rect
    largo = max(12, w // 5)
    for (px, py, dx, dy) in ((x, y, 1, 1), (x + w, y, -1, 1), (x, y + h, 1, -1), (x + w, y + h, -1, -1)):
        cv2.line(img, (px, py), (px + dx * largo, py), color, grosor + 1, cv2.LINE_AA)
        cv2.line(img, (px, py), (px, py + dy * largo), color, grosor + 1, cv2.LINE_AA)
    cv2.rectangle(img, (x, y), (x + w, y + h), color, 1, cv2.LINE_AA)


def ojos(img, lista_ojos, color=AMARILLO):
    for (ex, ey, ew, eh) in lista_ojos:
        cv2.circle(img, (ex + ew // 2, ey + eh // 2), max(3, ew // 4), color, 1, cv2.LINE_AA)


def barra_progreso(img, progreso, texto):
    """Barra inferior usada durante la captura de muestras de un rostro."""
    alto, ancho = img.shape[:2]
    margen = 16
    y = alto - 28
    cv2.rectangle(img, (margen, y), (ancho - margen, y + 14), NEGRO, cv2.FILLED)
    lleno = int((ancho - 2 * margen) * max(0.0, min(1.0, progreso)))
    cv2.rectangle(img, (margen, y), (margen + lleno, y + 14), AZUL, cv2.FILLED)
    etiqueta(img, texto, margen, y - 6, AZUL)


def hud(img, lineas):
    """Panel informativo en la esquina superior izquierda."""
    y = 22
    for linea in lineas:
        etiqueta(img, linea, 8, y, NEGRO, escala=0.5)
        y += 22


def barras_emociones(img, rect, emociones, ancho_barra=None):
    """
    Dibuja debajo del rostro las emociones más probables como barras horizontales.
    ``emociones`` es una lista de ``(nombre, probabilidad 0-1, color BGR)``.
    """
    x, y, w, h = rect
    alto_img, ancho_img = img.shape[:2]
    ancho_barra = ancho_barra or max(110, w)
    y0 = y + h + 8
    if y0 + 16 * len(emociones) > alto_img:  # sin espacio debajo: se dibuja dentro del rostro
        y0 = max(0, y + h - 16 * len(emociones) - 4)
    x = max(0, min(x, ancho_img - ancho_barra))
    for i, (nombre, prob, color) in enumerate(emociones):
        yy = y0 + i * 16
        cv2.rectangle(img, (x, yy), (x + ancho_barra, yy + 13), NEGRO, cv2.FILLED)
        cv2.rectangle(img, (x, yy), (x + int(ancho_barra * max(0.0, min(1.0, prob))), yy + 13), color, cv2.FILLED)
        texto = texto_ascii(f'{nombre} {prob * 100:.0f}%')
        cv2.putText(img, texto, (x + 3, yy + 10), FUENTE, 0.38, NEGRO, 2, cv2.LINE_AA)  # contorno
        cv2.putText(img, texto, (x + 3, yy + 10), FUENTE, 0.38, BLANCO, 1, cv2.LINE_AA)
