# some more imports :)...

import customtkinter as ctk
from PIL import Image
import os

class splash_screen:
   # Create a loading screen
    def __init__(self,root):
        # 1. Create Splash Screen Window
        self.splash = ctk.CTkToplevel(root)
        self.splash.title("Loading OmniGrad...")
        
        
        self.splash.overrideredirect(True)# makes it borderless
        
        # adjusting its location
        w, h = 550, 350
        sx = (self.splash.winfo_screenwidth() - w) // 2
        sy = (self.splash.winfo_screenheight() - h) // 2
        self.splash.geometry(f"{w}x{h}+{sx}+{sy}")
        
        # Splash UI Styling
        splash_frame = ctk.CTkFrame(self.splash, fg_color="#000000", border_width=2, border_color="#8B1E2D")
        splash_frame.pack(fill="both", expand=True)

        pil_image=Image.open(os.path.join(os.path.dirname(__file__),'omnigrad_logo.png'))# add custom logo
        logo_size=(150,150)
        logo=ctk.CTkImage(light_image=pil_image, dark_image=pil_image, size=logo_size)
        
        lbl_logo = ctk.CTkLabel(splash_frame, text="", image=logo)
        lbl_logo.image=logo
        lbl_logo.pack(pady=(60, 10))
        
        lbl_title = ctk.CTkLabel(splash_frame, text="OMNIGRAD", font=("Segoe UI", 28, "bold"), text_color="white")
        lbl_title.pack()
        
        lbl_load = ctk.CTkLabel(splash_frame, text="Pre-compiling neural nodes...", font=("Segoe UI", 13), text_color="#8F8F8F")
        lbl_load.pack(pady=20)
        
        # progress loading bar for splash
        self.splash_bar = ctk.CTkProgressBar(splash_frame, width=300, progress_color="#8B1E2D", fg_color="#303030")
        self.splash_bar.pack()
        self.splash_bar.configure(mode="indefinite")
        self.splash_bar.start()


