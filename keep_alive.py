"""
Mantiene despierta la app de Boris en Streamlit Cloud.

Abre la app en un navegador real (headless). Si ve el cartel de
"This app has gone to sleep", toca el botón y espera a que cargue el chat.
Si el chat no aparece a tiempo, termina con error para que GitHub te avise.
"""
import os
import sys
import time

from playwright.sync_api import sync_playwright

URL = os.environ.get(
    "APP_URL",
    "https://chatbotborisgit-txsap7uqcdfdunwmeo9ppi.streamlit.app/",
)
BOTON_DESPERTAR = "Yes, get this app back up!"
CHAT_INPUT = '[data-testid="stChatInput"]'
ESPERA_MAX_SEG = 240  # tiempo máximo esperando que cargue el chat


def buscar_boton(page):
    """Busca el botón de despertar en la página y en sus iframes."""
    for frame in page.frames:
        try:
            boton = frame.get_by_role("button", name=BOTON_DESPERTAR)
            if boton.count() > 0:
                return boton.first
        except Exception:
            pass
    return None


def chat_cargado(page):
    """True si el input del chat aparece en la página o en algún iframe."""
    for frame in page.frames:
        try:
            if frame.locator(CHAT_INPUT).count() > 0:
                return True
        except Exception:
            pass
    return False


def main():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        print(f"Abriendo {URL}")
        page.goto(URL, wait_until="domcontentloaded", timeout=60_000)
        time.sleep(8)  # margen para que renderice el cartel o la app

        boton = buscar_boton(page)
        if boton is not None:
            print("La app estaba dormida. Tocando el botón para despertarla...")
            boton.click()
        else:
            print("No hay cartel de sueño (la app ya estaba despierta o cargando).")

        inicio = time.time()
        while time.time() - inicio < ESPERA_MAX_SEG:
            if chat_cargado(page):
                print(f"OK: el chat de Boris cargó en {int(time.time() - inicio)}s.")
                browser.close()
                return 0
            time.sleep(5)

        print("ERROR: el chat no cargó a tiempo.")
        page.screenshot(path="error.png", full_page=True)
        browser.close()
        return 1


if __name__ == "__main__":
    sys.exit(main())