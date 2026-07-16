#not too many imports...

import customtkinter as ctk
from neural import MLP
from engine import visual
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image , ImageTk
import os
import sys


ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("dark-blue")


class ui_dis:
    # To create the skeleton ie that button,slider,frames etc of the app

    def __init__(self):
        self.app = ctk.CTk()
        icon_path=os.path.join(os.path.dirname(__file__),'omnigrad_logo.png')
        self.draw=visual()

        if os.path.exists(icon_path) and sys.platform !='darwin':
            # icon will be seen for win/linux no mac :(
            try:
               self.app.wm_iconbitmap()
               img = ImageTk.PhotoImage(Image.open(icon_path))
               self.app.iconphoto(True,img)
                
            except:
                pass

        window_width=1400
        window_height=850
        screen_height=self.app.winfo_screenheight()
        screen_width=self.app.winfo_screenwidth()
        x=(screen_width - window_width)//2
        y=(screen_height - window_height)//4
        self.app.geometry(f'{window_width}x{window_height}+{x}+{y}')
        self.app.title("OmniGrad")


        # ================= COLORS ================= #
        BG = "#121212"
        PANEL = "#1E1B1B"
        CARD = "#222222"
        BOX = "#303030"
        ACCENT = "#8B1E2D"
        HOVER = "#A52A3A"
        BORDER = "#3B3B3B"

        # ================= HEADER ================= #

        header = ctk.CTkFrame( self.app, height=60, 
                              fg_color="#181818", corner_radius=0 )
        header.pack(fill="x",pady=(15,2))
        header.pack_propagate(False)

        left_header = ctk.CTkFrame( header, fg_color="transparent" )
        left_header.pack(side="left", padx=20)

        self.logo = ctk.CTkLabel( left_header, 
                                 text="⬢", 
                                 text_color=ACCENT, 
                                 font=("Segoe UI", 26) )
        self.logo.pack(side="left", padx=(0, 10))

        # ================= TITLE ================= #
        title = ctk.CTkLabel(
            left_header,
            text="OMNIGRAD",
            font=("Segoe UI", 26, "bold"),
            text_color="white"
        )
        title.pack(anchor="w")

        brand = ctk.CTkFrame(
                left_header,
                fg_color="transparent"
            )
        brand.pack(side="left")

        subtitle = ctk.CTkLabel(
                    brand,
                    text="Every node, a variable. Every path, your choice.",
                    font=("Segoe UI",18),
                    text_color="#8F8F8F"
                                            )
        subtitle.pack(anchor="w")

        # ================= MAIN ================= #
        main = ctk.CTkFrame(
            self.app,
            fg_color=BG
        )
        main.pack(fill="both", expand=True, padx=25, pady=25)

        # ================= PANELS ================= #
        left = ctk.CTkFrame(
            main,
            fg_color=PANEL,
            corner_radius=18
        )

        right = ctk.CTkFrame(
            main,
            fg_color=PANEL,
            corner_radius=18
        )

        left.pack(side="left", fill="both", padx=(0, 20))

        
        right.pack(side="right", fill="both", expand=True)

        #creates a frame for the buttons graph,flowchar and output to fit in
        self.right_nav_bar = ctk.CTkFrame(master=right, fg_color="transparent")
        self.right_nav_bar.pack(fill="x", padx=20, pady=(15, 5))

        self.right_nav_bar.grid_columnconfigure(0, weight=1)
        self.right_nav_bar.grid_columnconfigure(1, weight=1)
        self.right_nav_bar.grid_columnconfigure(2, weight=1)

        # creates a frame for graph,flowchart and output
        self.view_output_frame = ctk.CTkFrame(master=right, fg_color="transparent")
        self.view_flowchart_frame = ctk.CTkFrame(master=right, fg_color="transparent", 
                                                 border_color=ACCENT,border_width=1,corner_radius=8)
        self.view_graph_frame = ctk.CTkFrame(master=right, fg_color="transparent")

        self.view_output_frame.pack(fill="both", expand=True)
        

        def show_output_layer():
            # shows output layout
            self.view_flowchart_frame.pack_forget()
            self.view_graph_frame.pack_forget()
            self.view_output_frame.pack(fill="both", expand=True)
            self.btn_nav_out.configure(fg_color=ACCENT, hover_color=HOVER)
            self.btn_nav_flow.configure(fg_color="#1E1B1B", hover_color="#2B2B2B")
            self.btn_nav_graph.configure(fg_color="#1E1B1B", hover_color="#2B2B2B")

        def show_flowchart_layer():
            # shows flowchart layout
            self.view_output_frame.pack_forget()
            self.view_graph_frame.pack_forget()
            self.view_flowchart_frame.pack(side="right", padx=30, pady=30, fill="both", expand=True)
            self.view_flowchart_frame.pack_propagate(False)
            self.btn_nav_flow.configure(fg_color=ACCENT, hover_color=HOVER)
            self.btn_nav_out.configure(fg_color="#1E1B1B", hover_color="#2B2B2B")
            self.btn_nav_graph.configure(fg_color="#1E1B1B", hover_color="#2B2B2B")
            

        def show_graph_layer():
            # shows the graph layout
            self.view_output_frame.pack_forget()
            self.view_flowchart_frame.pack_forget()
            self.view_graph_frame.pack(fill="both", expand=True)
            self.btn_nav_graph.configure(fg_color=ACCENT, hover_color=HOVER)
            self.btn_nav_out.configure(fg_color="#1E1B1B", hover_color="#2B2B2B")
            self.btn_nav_flow.configure(fg_color="#1E1B1B", hover_color="#2B2B2B")

        # creates a top frame and button to switch between output,flowchart and graph
        self.btn_nav_out = ctk.CTkButton(
            master=self.right_nav_bar, text="Generated Text", height=38, font=("Segoe UI", 14, "bold"),
            fg_color=ACCENT, text_color="white", hover_color=HOVER, corner_radius=10, command=show_output_layer
        )
        self.btn_nav_out.grid(row=0, column=0, sticky="ew", padx=(0, 5))

        self.btn_nav_flow = ctk.CTkButton(
            master=self.right_nav_bar, text="Backpropagation Flowchart", height=38, font=("Segoe UI", 14, "bold"),
            fg_color="#1E1B1B", text_color="white", hover_color="#2B2B2B", corner_radius=10, command=show_flowchart_layer
        )
        self.btn_nav_flow.grid(row=0, column=1, sticky="ew", padx=5)

        self.btn_nav_graph = ctk.CTkButton(
            master=self.right_nav_bar, text="Epoch vs Loss Graph", height=38, font=("Segoe UI", 14, "bold"),
            fg_color="#1E1B1B", text_color="white", hover_color="#2B2B2B", corner_radius=10, command=show_graph_layer
        )
        self.btn_nav_graph.grid(row=0, column=2, sticky="ew", padx=(5, 0))

        # creates output box where out put will be shown as logs
        self.output_box = ctk.CTkTextbox(
            master=self.view_output_frame, fg_color="#303030", border_width=1, border_color=ACCENT, corner_radius=12, font=("Consolas", 15)
        )
        self.output_box.pack(fill="both", expand=True, padx=20, pady=5)

        self.output_box.insert("1.0", "Initializing Model...\nWaiting for execution.\n\n")

        # creates the canvas for flowchart resizing 
        self.canvas = ctk.CTkCanvas(self.view_flowchart_frame, bg='#E5E7EB',highlightthickness=0)
        self.canvas.pack(fill="both", expand=True, padx=20, pady=20)
        self.canvas_label = ctk.CTkLabel(self.canvas, text="Tree visualization will process here after run.", 
                                         text_color="#0E0D0D",corner_radius=8)
        self.canvas_label.pack(expand=True, fill="both", padx=20, pady=20)
        self.canvas_label.forget()

        # creates the layout of the matplotlib graph container
        self.fig, self.ax = plt.subplots(figsize=(6, 4), facecolor="#1E1B1B")
        self.ax.set_facecolor("#121212") 
        

        self.ax.tick_params(axis='both',which='both',colors='#8F8F8F', labelsize=10)
        self.ax.xaxis.label.set_color('#8F8F8F')
        self.ax.yaxis.label.set_color('#8F8F8F')
        for spine in self.ax.spines.values():
            spine.set_color('#3B3B3B')

        self.graph_canvas = FigureCanvasTkAgg(self.fig, master=self.view_graph_frame)
        self.graph_canvas.get_tk_widget().pack(fill="both", expand=True, padx=20, pady=20)
        
        self.ax.grid(True, color="#252525", linestyle="--")
        self.ax.set_title("No training data available. Run engine first.", color="#7F7F7F", fontsize=12)
        self.graph_canvas.draw()

        # frame where download,generate and clear will be fit
        button_frame = ctk.CTkFrame(right, fg_color='transparent')
        button_frame.pack(side="bottom", fill="x", pady=10, padx=20)

        # create the generate button to start the trianing loop
        self.generate_btn = ctk.CTkButton(button_frame, text="Generate", height=44, fg_color=ACCENT, hover_color=HOVER, corner_radius=10, font=("Segoe UI", 15, "bold"))
        self.generate_btn.pack(side="left", fill="x", expand=True, padx=(0, 15))

        # Creates the clear button to clear the output logs 
        self.clear_btn = ctk.CTkButton(button_frame, text="Clear", height=44, fg_color="#3B3B3B", hover_color="#4A4A4A", 
                                       corner_radius=10, font=("Segoe UI", 15), width=110)
        self.clear_btn.pack(side="left", padx=12)

        # creates the download button to download flowchart 
        self.download_btn = ctk.CTkButton(button_frame, text="Download", height=44, fg_color="#3B3B3B", hover_color="#4A4A4A", corner_radius=10, font=("Segoe UI", 15), width=130)
        self.download_btn.pack(side="left", padx=12)


        right_header = ctk.CTkFrame(header, fg_color="transparent")
        right_header.pack(side="right", padx=20, fill="y")
        
        # creates the progress bar on top right to see how much the model has trained
        self.status_label = ctk.CTkLabel(
            right_header, text="Status: Idle", font=("Segoe UI", 13, "bold"), text_color="#8F8F8F"
        )
        self.status_label.pack(side="top", anchor="e", pady=(5, 2))

        self.progress_bar = ctk.CTkProgressBar(
            right_header, width=300, height=8, progress_color=ACCENT, fg_color=BOX
        )
        self.progress_bar.pack(side="bottom", pady=(0, 12))
        self.progress_bar.set(0)

        # ================= INPUTS ================= #
        labels = [
            "Target Loss".title(),
            "No of Inputs per set".title(),
            "Input".title(),
            "Learning Rate".title(),
            "No of neurons in each layer".title(),
            "Expected Output".title(),
            "No of Loops".title(),
            "Type of Activation and Loss".title(),
        ]

        self.inputs = []
        self.labels =[]

        for r in range(4):
            # configure the left frame 
            left.grid_rowconfigure(r, weight=1)

        left.grid_columnconfigure(0, weight=1)
        left.grid_columnconfigure(1, weight=1)

        for i in range(8):
            # create text box their frames and label using loop
            row = i % 4
            col = i // 4

            card = ctk.CTkFrame(
                left,
                fg_color=CARD,
                corner_radius=14,
                border_width=1,
                border_color="#424242"
            )

            card.grid(
                row=row,
                column=col,
                padx=18,
                pady=16,
                sticky="nsew"
            )

            label = ctk.CTkLabel(
                card,
                text=labels[i],
                text_color=ACCENT,
                anchor="w",
                font=("Segoe UI", 15, "bold")
            )

            self.labels.append(label)
            label.pack(anchor="w", padx=12, pady=(10, 6))

            if i==0:
                self.acc_entry = ctk.CTkEntry(
                    card,
                    height=38,
                    placeholder_text="e.g. 0.01",
                    font=("Segoe UI", 14)
                )
                self.acc_entry.insert(0, "0.0")
                self.acc_entry.pack(
                    padx=12,
                    pady=(0, 10),
                    fill="x"
                )
                switch_frame = ctk.CTkFrame(card, fg_color="transparent")
                switch_frame.pack(fill="x", padx=12, pady=(0, 10))

                label = ctk.CTkLabel(
                    switch_frame,
                    text="Reuse Existing Model        ",
                    font=("Segoe UI", 15,'bold'),
                    text_color=ACCENT
                )
                label.pack(side="left",pady=14)


                self.reuse_switch = ctk.CTkSwitch(
                    switch_frame,
                    progress_color=ACCENT,
                    text=""
                )
                self.reuse_switch.pack(side="right",pady=14)

            elif i==7:
                act_menu = ctk.CTkOptionMenu(
                    card,
                    width=260,
                    values=[
                        "Tanh",
                        "RELU",
                        "Leaky RELU",
                        "Sigmoid",
                        "Swish",
                        "GELU"
                    ],
                    fg_color=ACCENT,
                    button_color=ACCENT,
                    button_hover_color=HOVER,
                    dropdown_fg_color=CARD,
                    dropdown_hover_color="#353535",
                    dropdown_text_color="white",
                    font=("Segoe UI", 14,'bold'),
                    height=38,
                )

                act_menu.pack(padx=12, pady=(0, 12), fill="x")      

                loss_menu = ctk.CTkOptionMenu(
                    card,
                    width=260,
                    values=[
                        "Mean square Error",
                        "Mean Absolute Error"
                    ],
                    fg_color=ACCENT,
                    button_color=ACCENT,
                    button_hover_color=HOVER,
                    dropdown_fg_color=CARD,
                    dropdown_hover_color="#353535",
                    dropdown_text_color="white",
                    font=("Segoe UI", 14,'bold'),
                    height=38
                )

                loss_menu.pack(padx=12, pady=(0, 12), fill="x")

                self.act_menu = act_menu
                self.loss_menu = loss_menu

            elif i==3:
                    slider = ctk.CTkSlider(
                                card,
                                from_=0.001,
                                to=0.1,
                                width=260,

                                button_color=ACCENT,        
                                button_hover_color=HOVER,    
                                progress_color=ACCENT,       
                                fg_color=BOX,    

                                command=lambda val: self.lr_label.configure(text=f"{val:.3f}")
                            )
                    slider.pack(padx=12, pady=(19, 19), fill="x")   
                    self.slider=slider
                    self.lr_label = ctk.CTkLabel(card, text= "0.05", 
                                                 font=("Segoe UI", 16,'bold'),
                                                 text_color="#A0A0A0")
                    self.lr_label.pack(padx=12, pady=(0, 2))        

            else:
                # create the text box using loop
                box = ctk.CTkTextbox(
                    card,
                    width=260,
                    height=95,
                    fg_color=BOX,
                    border_width=1,
                    border_color="#4A4A4A",
                    corner_radius=10,
                    font=("Segoe UI", 14)
                )

                box.pack(padx=12, pady=(0, 12), fill="both")

                self.inputs.append(box)


    def update_graph(self,loss,epochs):
        # To make the stylin of the matplotlib graph 

        self.ax.clear()
        self.ax.set_facecolor("#121212")
        self.ax.grid(True, color="#252525", linestyle="--")
        self.ax.set_yscale('log')
        self.ax.plot(epochs,loss,color="#8B1E2D", linewidth=2.5, label="Training Loss")
        self.ax.set_title("Loss Convergence Curve", color="white", fontsize=14, fontweight="bold", pad=15)
        self.ax.set_xlabel("Epochs / Loops", fontsize=12,color="#8F8F8F", labelpad=8)
        self.ax.set_ylabel("Loss Value (logarathmic)", fontsize=12,color="#8F8F8F", labelpad=8)
        self.fig.tight_layout()
        self.graph_canvas.draw()

    def run(self):
        #This funciton is made so all internal loops/function running will come to stop
        self.app.protocol("WM_DELETE_WINDOW", self._internal_close)
        self.app.mainloop()

    def _internal_close(self):
        import sys
        self.app.withdraw()
        sys.exit(0)



