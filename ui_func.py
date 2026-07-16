#Never enough imports....

from ui_init import ui_dis
from neural import MLP
from loading_screen import splash_screen
from tkinter import filedialog
from CTkToolTip import CTkToolTip as ctktool
import textwrap
from PIL import Image , ImageTk
import threading 
import time

class ui_func:
        
        def __init__(self):
# just some init stuff
            
            self.exe = ui_dis() # creates the skeleton object for display
            self.exe.generate_btn.configure(command=self.worker_thread)# starts the worker thread when genrate is clicked
            self.nn=None
            self.exe.output_box.tag_config("error", foreground="#F1C40F")

            self.zoom_factor=1.0
            handlers=self.flowchart_resizer()

            # Binding the mouse buttons to their even for handling the flowchart resizing
            self.exe.canvas.bind('<ButtonPress-1>',handlers['press'])
            self.exe.canvas.bind('<B1-Motion>',handlers['motion'])
            self.exe.canvas.bind('<MouseWheel>',handlers['zoom'])
            self.exe.canvas.bind('<Button-4>',handlers['linux_zoom'])
            self.exe.canvas.bind('<Button-5>',handlers['linux_zoom'])

            self.t_inputs = {}
            self.o_box = self.exe.output_box
            self.exe.clear_btn.configure(command=self.clear)
            self.o_box.configure(state='disabled')
            self.cancel_requested= False
            self.load_screen = None
            
            
# since we made the input boxes using a loop we save the input in dict with their key for easier access
            label=['no_inp','input',
                   'ls_neuron',
                   'exp_out','no_loop']
            
            for i in range(len(self.exe.inputs)):
                 self.t_inputs[label[i]]=self.exe.inputs[i]
                 
            #--------TOOLTIP-----------#
# Create Tool tip for all the input boxes
            ctktool(
                self.t_inputs["no_inp"],
                message=textwrap.dedent(
                    """\
                    Number of inputs per dataset.
                    Example: 3
                    Type: Positive integer"""
                ),
                delay=0.1,
            )

            ctktool(
                self.t_inputs["input"],
                message=textwrap.dedent(
                    """\
                    Enter your input dataset (one row per set).
                    Example (for 3 inputs):
                    1, 2, 3
                    4, 5, 6
                    Note: The items in each row must match the 'Number of inputs' value.
                    Type: Comma-separated floats"""
                ),
                delay=0.1,
            )

            ctktool(
                self.t_inputs["ls_neuron"],
                message=textwrap.dedent(
                    """\
                    Define the hidden layers and their sizes.
                    Example: 3,4,5,2,1 (5 hidden layers with 3, 4, 5, 2, and 1 neurons)
                    Type: Comma-separated integers"""
                ),
                delay=0.1,
            )

            ctktool(
                self.t_inputs["exp_out"],
                message=textwrap.dedent(
                    """\
                    Expected output for each row in your input dataset.
                    Example (for 2 rows):
                    0.34
                    0.78
                    Note: Ensure values fit the range of your chosen activation function.
                    Type: One float value per line"""
                ),
                delay=0.1,
            )

            ctktool(
                self.t_inputs["no_loop"],
                message=textwrap.dedent(
                    """\
                    Total training iterations (epochs) for the model.
                    Example: 100
                    Note: Very high values will significantly slow down execution.
                    Type: Positive integer"""
                ),
                delay=0.1,
            )
            ctktool(
                    self.exe.slider,  
                    message=textwrap.dedent(
                       """\
        Controls the step size during training.
        * High: Fast but unstable.
        * Low: Stable but very slow."""
                    ),
                    delay=0.1,
                )
        def error_handler(self,message):
            self.o_box.configure(state='normal')
            self.o_box.insert('end',message,'error')
            self.o_box.configure(state="disabled")



        def worker_thread(self):
            # Create a worker thread to handle the Math so the UI doesnt get frozen

            #----------EXCEPTION HANDLING----------#
            try:
                exp_loss = float(self.exe.acc_entry.get())
                if exp_loss<0:
                    self.error_handler('\n>> Negative Value was entered "Target Loss" field \n'
                                            'Please Enter Positive Value\n\n')
                    return

            except ValueError:
                 self.error_handler('\n>> No value or incorrect value was given for ''Target Loss''field\n'
                                            'Please Enter Valid Value\n\n')
                 return
                 
            reuse = self.exe.reuse_switch.get()

            try: # handling exception in case of empty or incorrect value given by user
                input_size = int(self.t_inputs['no_inp'].get('1.0', 'end').strip())# get input size
                if input_size <= 0:
                    self.error_handler('\n>> No value or incorrect value was given for "No of Inputs per set" field\n'
                                            'Please Enter Valid Value\n\n')
                    return

            except ValueError:
                self.error_handler('\n>> No value or incorrect value was given for "No of Inputs per set" field\n'
                                            'Please Enter Valid Value\n\n')
                return
            
            try:# handling exception in case of empty or incorrect value given by user
                layers = [int(i) for i in self.t_inputs['ls_neuron'].get('1.0', 'end').strip().split(',')]# get layer list
                if any(layer <=0 for layer in layers):
                    self.error_handler('\n>> No value or incorrect value was given for "No of Neurons in each layer"" field\n'
                                            'Please Enter Valid Value\n\n')
                    return
                if layers[-1] !=1: # making sure the last layer always has 1 neuron
                    layers+=[1]
                    
            except ValueError:
                self.error_handler('\n>> No value or incorrect value was given for "No of Neurons in each layer"" field\n'
                                            'Please Enter Valid Value\n\n')
                return

            try:# handling exception in case of empty or incorrect value given by user
                raw_inputs = self.t_inputs['input'].get('1.0','end').strip().split('\n')# get inputs
                xs = [[float(i) for i in line.split(',')] for line in raw_inputs if line.strip()]
                check_x=2/float(len(xs))
                if any(len(row)!=input_size for row in xs):
                    self.error_handler(
                        '>> Invalid input data.\n'
                        'Each row in the "Input" field must contain the same number of values '
                        'as specified in "No. of Inputs per Set".\n\n'
                    )
                    return
            except (ZeroDivisionError,ValueError):
                self.error_handler('\n>> No value or incorrect value was given for ''Input'' field\n'
                                            'Please Enter Valid Value\n\n')
                return
                 
            try:# handling exception in case of empty or incorrect value given by user
                raw_targets = self.t_inputs['exp_out'].get('1.0','end').strip().split('\n')# get expected out
                ys = [float(i) for i in raw_targets if i.strip()]
                check_y=2/float(len(ys))
                if len(ys) != len(xs):
                        self.error_handler(
                            '>> Dataset size mismatch.\n'
                            'The number of expected outputs must match the number of input rows.\n\n'
                        )
                        return
                
            except (ZeroDivisionError,ValueError):     
                self.error_handler('\n>> No value or incorrect value was given for ''Expected Output'' field\n'
                                            'Please Enter Valid Value\n\n')
                return

            try:# handling exception in case of empty or incorrect value given by user
                n = int(self.t_inputs['no_loop'].get('1.0','end').strip()) # no of loops
                if n<=0 :
                    self.error_handler('\n>> No value or incorrect value was given for ''No of Loops'' field\n'
                                            'Please Enter Valid Value\n\n')    
                    return               
            except ValueError:
                self.error_handler('\n>> No value or incorrect value was given for ''No of Loops'' field\n'
                                            'Please Enter Valid Value\n\n')
                return
                 

            current_config = (input_size, tuple(layers))

            if reuse:
                # is reusing the model the previous architeture shd be same , handles exceptions
                if self.nn is None:
                    
                   self.nn=MLP(input_size,layers)
                   self.model_config=current_config

                elif current_config != self.model_config: # if model architecture is not same as before make new architecture
                    self.exe.status_label.configure(text='Error')
                    self.o_box.configure(state="normal")
                    self.o_box.insert(
                        "end",
                        "Cannot reuse the existing MLP.\n"
                        "The model architecture has been changed.\n"
                        "Please disable 'Reuse Existing Model' or restore the previous architecture.\n\n",
                        'error'
                    )
                    self.o_box.configure(state="disabled")
                    return
            else:
                self.nn= MLP(input_size,layers)
                self.model_config=current_config    

            lr=float(self.exe.slider.get()) # get learning rate 
            loss_list=[]
            epoch_list=[]
            

            self.cancel_requested= False #initializing cancel button 
            worker=threading.Thread(target=self.worker_processing,
                                    args=(self.nn,exp_loss,xs,ys,n,lr,loss_list,epoch_list))
            worker.daemon=True # makes it so if main thread is close the worked thread also will halted
            worker.start()

        def worker_processing(self,nn,exp_loss,xs,ys,n,lr,loss_list,epoch_list):
            # Handle the math in In the worker thread 

            start_time=time.time()# starts the timer
            ypred=None
            loss=None
            initial_pred=None

            self.exe.status_label.configure(text='Status : Training...')
            self.exe.generate_btn.configure(text='Cancel',command=self._cancel)

            for k in range(n):

                if self.cancel_requested:
                     break
                
                ypred = [self.nn(x,act_type=self.exe.act_menu.get()) for x in xs]# creating the model with given architecture

                if k==0:
                     initial_pred=ypred
                     
                
                if self.exe.loss_menu.get() == "Mean square Error": # mean sqaure error definition
                    MSE=sum([(y_true-yout)**2 for y_true,yout in zip(ys,ypred)])/len(ys) if type(ys) == list else 1
                    loss =MSE
                elif self.exe.loss_menu.get() == "Mean Absolute Error": # mean absolute error definition
                    MAE=sum([(y_true-yout).abs_val() for y_true,yout in zip(ys,ypred)])/len(ys) if type(ys) == list else 1
                    loss= MAE
                
                loss_list.append(loss.data if hasattr(loss,'data') else loss)
                epoch_list.append(k+1)

                for p in self.nn.parameters():
                    p.grad=0
                loss.backward() # we call backward on loss 

                if loss.data <= exp_loss:
                     break

                for p in self.nn.parameters():
                    p.data += -lr * p.grad
                progress= (k+1)/n

                self.exe.app.after(0,lambda p=progress: self.exe.progress_bar.set(p))# update the progress bar
                if k % 100==0:# updates the status bar
                     self.exe.app.after(0,lambda:self.exe.status_label.configure(
                                                  text=f"Status: Training...\nEpoch: {k+1:,} / {n:,}"
                                            ))
            elapsed=time.time()-start_time # calculates the total time taken to finish training
                                                        
            self.exe.app.after(0,lambda:self.training_finish(loss,loss_list,initial_pred,ypred,epoch_list,elapsed))
            # we use after() to safely push it to main thread 

        def training_finish(self,loss,loss_list,initial_pred,ypred,epoch_list,elapsed_time):
            #to configure the output ,flowchart,graph,reset the progress bar etc 
            # after completion of training loop
            self.o_box.configure(state='normal')
            self.o_box.insert("end", "\n" + "─" * 40 + "\n")
            self.o_box.insert("end", "Training Summary\n\n")
            # Runtime
            self.o_box.insert("end", "Runtime\n")
            self.o_box.insert("end", f"    {elapsed_time:.2f} seconds\n\n")

            # Loss
            self.o_box.insert("end", "Loss\n")
            self.o_box.insert("end", f"    Initial : {loss_list[0]:.8f}\n")
            self.o_box.insert("end", f"    Final   : {loss.data:.8f}\n\n")

            # Predictions
            self.o_box.insert("end", "Predictions\n\n")

            self.o_box.insert("end", "Initial\n")
            for i, pred in enumerate(initial_pred, start=1):
                self.o_box.insert("end", f"    Output {i:<2}: {pred.data:.6f}\n")

            self.o_box.insert("end", "\nFinal\n")
            for i, pred in enumerate(ypred, start=1):
                self.o_box.insert("end", f"    Output {i:<2}: {pred.data:.6f}\n")

            self.o_box.see("end")
            self.o_box.configure(state='disabled')

            #reinitialize for next model
            self.exe.generate_btn.configure(text='Generate',command=self.worker_thread)
            self.exe.status_label.configure(text='Status : Idle')
            self.exe.progress_bar.set(0)
            self.exe.draw.draw_dot(loss)
            self.flowchart_resizer() # update the flowchart 
            self.exe.download_btn.configure(command=self.download)
            self.exe.update_graph(loss_list,epoch_list)
            
        
        def clear(self):
             # To clear the previous output and error logs

             self.o_box.configure(state='normal')
             self.o_box.delete('1.0','end')
             self.o_box.insert("1.0", "Initializing Model...\nWaiting for execution.\n\n")
             self.o_box.configure(state='disabled')

        def download(self):
            # To download the backprop flowchart

            # only svg as the image will be too big and to have good dpi
            save_path=filedialog.asksaveasfilename(title='Save the Flowchart as...',
                                                    defaultextension='.svg',
                                                    filetypes=[('SVG files','*.svg')])
            if not save_path:
                 return
            else:
                try:
                    self.exe.draw.dot.format='svg'
                    self.exe.draw.dot.render(outfile=save_path,cleanup=True)
                except FileNotFoundError:
                     return
                      
        def _cancel(self):
             # to exit the training loop
             self.cancel_requested= True
             

        def flowchart_resizer(self):
            # this function is to pan around zoom in/out of the flowchart display/we use CTkcanvas
            min_zoom=1.0
            max_zoom=20.0
            
            
            self.user_zoom=3.0

            def update_image():
                 # updates image everytime a mouse event takes place 
                 if not hasattr(self.exe.draw,'pil_image'):
                      self.exe.canvas_label.pack(expand=True, fill="both", padx=20, pady=20)
                      return
                 
                 self.exe.canvas_label.forget()
                 

                 canvas_w=self.exe.canvas.winfo_width()
                 canvas_h=self.exe.canvas.winfo_height()
 
                 base_img = self.exe.draw.pil_image 

                 img_w=base_img.width
                 img_h=base_img.height

                 fit_factor=min(canvas_w/img_w,canvas_h/img_h) # to display the whole image at the start

                 self.zoom_factor=fit_factor * self.user_zoom
            
                 new_w=int(self.zoom_factor * base_img.width)
                 new_h=int(self.zoom_factor * base_img.height)

                 if new_w < 1 or new_h < 1: # handles exception in case fit_factor retunr very small number
                      return
                 
                 resize_img=base_img.resize((new_w,new_h),Image.Resampling.BICUBIC)
                 self.tk_img=ImageTk.PhotoImage(resize_img)

                 self.exe.canvas.delete('all')
                 self.exe.canvas.create_image(0,0,anchor='nw',image=self.tk_img)
                 self.exe.canvas.config(scrollregion=(0,0,new_w,new_h))
                 self.exe.canvas.image=self.tk_img
                 

            def start_pan(event):
                 #save the mouse click point
                 self.exe.canvas.scan_mark(event.x,event.y)

            def pan_img(event):
                 # pan the image
                 self.exe.canvas.scan_dragto(event.x,event.y,gain=1)
            
            def zoom(event):
                # zooming for windows/mac
                if event.delta > 0:
                      self.user_zoom=min(max_zoom,self.user_zoom * 1.1)
                else :
                      self.user_zoom=max(min_zoom,self.user_zoom / 1.1)

                update_image()

            def zoom_linux(event):
                # zooming for linux 
                if event.num == 4:
                      self.user_zoom=min(max_zoom,self.user_zoom + 0.1)
                elif event.num ==5:
                     self.user_zoom=max(min_zoom,self.user_zoom - 0.1)
                
                update_image()

            update_image()
            return {
                'press': start_pan,
                'motion': pan_img,
                'zoom': zoom,
                'linux_zoom': zoom_linux
            }

        def start_app(self):
            # boot the app up

            self.load_screen = splash_screen(self.exe.app)
            self.load_screen.splash.after(500,self.boot_sequence)
            self.load_screen.splash.mainloop()

        def boot_sequence(self):
            # Demolish the splash screen
            self.load_screen.splash.destroy()     

            self.exe.run()


            

app=ui_func()
app.start_app()
