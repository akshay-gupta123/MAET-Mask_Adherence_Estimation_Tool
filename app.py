import os
from flask import Flask, render_template,request, redirect, jsonify
from predict import get_prediction
from lime_pred import lime_result
import torch
import time
    
app = Flask(__name__)
@app.route('/',methods=['GET','POST'])
def upload_file():
    if request.method == 'POST':
        if "target" in request.form:
            target = request.form.get('target')
            f = open('./static/read.txt','r')
            num_mask, num_non_mask, level, model_img_address, total_pred, image_id = f.read().split(" ")[-7:][:6]
            
            if target == "":
                target = 0 
            if total_pred and not os.path.exists(f"./static/{model_img_address}_{target}_1.jpg"):
                lime_result(image_id,model_img_address,int(target))
            level = "Can't Say" if not total_pred else level
            out_bound = 0 if total_pred else 1
            return render_template('result.html',class_mask=num_mask, class_nmask=num_non_mask, class_pro=level, img_name=model_img_address, total_pred= total_pred, out_bound= out_bound, target= target)
            
        elif 'file' not in request.files:
            return redirect(request.url)
        
        else:
            file = request.files.get('file')
            if not file:
                return render_template('index.html')
            
            img_byte = file.read()
            
            image_id, num_mask, num_non_mask, level, model_img_address, total_pred = get_prediction(img_byte)
            if total_pred :
                lime_result(image_id,model_img_address)
            level = "Can't Say" if not total_pred else level
            out_bound = 0 if total_pred else 1
            
            f = open('./static/read.txt',"a") 
            string = f'{num_mask} {num_non_mask} {level} {model_img_address} {total_pred} {image_id} '
            f.write(string)
            f.close()
            
            return render_template('result.html',class_mask=num_mask, class_nmask=num_non_mask, class_pro=level, img_name=model_img_address, total_pred=total_pred, out_bound=out_bound, target=0)
    return render_template('index.html')
    
if __name__=='__main__':
    app.run(debug=True,port="5000")