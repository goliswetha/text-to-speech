import fitz
from transformers import VitsModel, AutoTokenizer
import torch
import soundfile as sf
from flask import Flask, render_template, request, send_file
import os
from werkzeug.utils import secure_filename
import pytesseract
from pytesseract import image_to_string
import numpy as np
import pytesseract

import os

from PIL import Image


pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def converting_pdf_into_images(file_path):
    with fitz.open(file_path) as doc :
            for i,page in enumerate(doc):
                page = doc.load_page(i)  
                pix = page.get_pixmap()
                output = f"instance/htmlfi/intermediate{i}.png"
                pix.save(output)



def converting_to_audio(file_paths):
    
    hindi_text = pytesseract.image_to_string(Image.open(file_paths), lang='hin') 
    print(hindi_text) 
            
    if hindi_text:
        model = VitsModel.from_pretrained("facebook/mms-tts-hin")
        tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-hin")
        inputs = tokenizer(hindi_text, return_tensors="pt")

        with torch.no_grad():
            output = model(**inputs).waveform
        return output,model
    else:
        return np.zeros(5),np.zeros(5)

app = Flask(__name__)

@app.route("/")
def upload_form():
    return render_template("index.html")  

@app.route("/uploader", methods=['POST', 'GET'])
def upload_file():
    if request.method == 'POST':
        uploaded_file = request.files['files']  
        if uploaded_file is None:
            return "Please select a file to upload."
        #To store filename and not corrupt server side 
        filename = secure_filename(uploaded_file.filename)
        checker=filename
        print(filename)
        #checing if the file is pdf or png
        x,y=checker.split(".")
        print(y)
        os.makedirs(os.path.join(app.instance_path, 'htmlinputfi'), exist_ok=True)
        os.makedirs(os.path.join(app.instance_path,'htmloutfi'),exist_ok=True)
        os.makedirs(os.path.join(app.instance_path,'htmlfi'),exist_ok=True)
        file_path = os.path.join(app.instance_path, 'htmlinputfi', filename)
        uploaded_file.save(file_path)
        if(y =='png'):
           output,model= converting_to_audio(file_path)
           if((output<=0).all()):
               return "No text found"
           #normalizing the output and usin its sample rate to avoid pitch changes
           output_np = output.squeeze().cpu().numpy()
           output_np = output_np / max(abs(output_np))
           sample_rate = model.config.sampling_rate
           
           audio_path=os.path.join(app.instance_path,'htmloutfi',f'{x}.wav')
           sf.write(audio_path,output_np,sample_rate)
           
        #generating a wav file with vague noise to avoid garbge values/cache 
        audio_path = os.path.join(app.instance_path, 'htmloutfi', f"{x}.wav")
        array = np.array([0,0.5,0.75, 1, 2])
        sf.write(audio_path,array,1)
        converting_pdf_into_images(file_path)
        open_file=fitz.open(file_path)
        for i in range(open_file.page_count):
            output,model=converting_to_audio(f'instance/htmlfi/intermediate{i}.png')

            if((output<=0).all()):
                return "no hindi text available"
            #normalizing the output and using its sample rate to avoid pitch changes
            output_np = output.squeeze().cpu().numpy()
            output_np = output_np / max(abs(output_np))
            sample_rate = model.config.sampling_rate
            
            
            
            demo,sample=sf.read(f'instance/htmloutfi/{x}.wav')
            #concatenating the audio files into one
            demo=np.array(demo,dtype=np.float32)
            output_np=np.array(output_np,dtype=np.float32)
            concate=np.concatenate((demo,output_np),axis=None)
            
            
            sf.write(audio_path, concate, sample_rate)  

    return send_file(audio_path,as_attachment=True)
       

    return "Something went wrong"

if __name__ == "__main__":
    app.run(port=8000, debug=True)