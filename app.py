from flask import *
import pickle

import numpy as np

app=Flask(__name__)



def check_msg(msg):
    vec=pickle.load(open("tfidfvect.pkl","rb"))
    lr=pickle.load(open("lreg.pkl","rb"))
    
    data=vec.transform([msg]).toarray()
    ans=lr.predict(data)
    return ans
    
    
    
    
@app.route("/")
def homepage():
    return render_template("home.html")

@app.route("/senddata",methods=["POST"])
def fetchdata():
    msg=request.form["t1"]
    
    ans=check_msg(msg)
    return render_template("display.html",data=ans)


if (__name__=="__main__"):
    app.run(debug=True)