from flask import Flask, jsonify, request,render_template
from functions import *
import base64
import matplotlib


UPLOAD_FOLDER = './uploads'
SAVE_FOLDER='../save/'
#personas_likes= [ {},{},{},{},{} ]
#pickle.dump(personas_likes, open(SAVE_FOLDER+"personas_likes.pkl", "wb"))
std_scale = pickle.load(open(SAVE_FOLDER+"kmeans_12.pkl", "rb"))

std_scale=pickle.load( open(SAVE_FOLDER+"std_scale.pkl", "rb"))
pca= pickle.load(open(SAVE_FOLDER+"pca.pkl", "rb")) 
data_sortie=pickle.load( open(SAVE_FOLDER+"data_sortie.pkl", "rb"))
names=["../thumbs/img"+str(i)+".png" for i in range(len(data_sortie)+1)]
X_std=pickle.load( open(SAVE_FOLDER+"X_std.pkl", "rb"))
personas_likes= pickle.load( open(SAVE_FOLDER+"personas_likes.pkl", "rb"))
labels_={}
centers_={}
kms_={}
for iclusters in [12,24,48]:
    labels_[iclusters]=pickle.load( open(SAVE_FOLDER+"labels_"+  str(iclusters)+".pkl", "rb"))
    centers_[iclusters]=pickle.load( open(SAVE_FOLDER+"centers_"+  str(iclusters)+".pkl", "rb"))
    kms_[iclusters]=pickle.load( open(SAVE_FOLDER+"kmeans_"+  str(iclusters)+".pkl", "rb"))
    
personas =[
   
     {"name":"Peter","img":567, "age":"23","criterias":"Aime les jeunes femmes blondes."},
         {"name":"Sophie","img":484, "age":"25","criterias":"Aime les hommes bruns avec de la barbe."},
   {  "name":"John","img":1415, "age":"48","criterias":"Aime les femmes d'origine asiatique pas trop jeunes."},
    {"name":"Carmelle","img":1081, "age":"27","criterias":"Aime les hommes à la peau mate ou foncée."},
 {"name":"Me","img":-1, "age":"?","criterias":"..."}
]    
    
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return render_template('index.tpl_html')


@app.route("/multi/<int:num1>/<int:num2>", methods=["GET"])
def get_muliply10(num1,num2):
    return jsonify({"result":num1*num2})
    
@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file1' not in request.files:
            return 'there is no file1 in form!'
        file1 = request.files['file1']
        path = os.path.join(app.config['UPLOAD_FOLDER'], file1.filename)
        file1.save(path)
        vectors,thumb_file=image_vectorize(path,std_scale,pca)
        labeled  =kms_[48].predict([vectors])
        igroup = labeled[0]
        bg=matplotlib.colors.rgb2hex(cm.rainbow(np.linspace(0,1,len(centers_[48])))[igroup])

        with open(thumb_file, "rb") as img_file:
            b64_string = base64.b64encode(img_file.read()).decode("utf-8")
        nearests = get_nearests_from(vectors,data_sortie,5)
        likeperson = [  base64.b64encode(open(names[i+1], "rb").read()).decode("utf-8")   for i in nearests]
        return  render_template('show_image.tpl_html',b64_string = b64_string,  n_likeperson = len(likeperson),likeperson=likeperson  ,igroup=igroup, bg=bg )
    return  render_template('upload.tpl_html', bg="#AAAAAAFF")

@app.route("/personas")
def choosepersonas():
    for i,p in enumerate(personas):
        if p["img"]==-1:
            file= "./static/userme.jpeg"
        else:
            file = names[p["img"]+1]
        personas[i]["src"]="data:image/png;base64,"+  base64.b64encode(open(file, "rb").read()).decode("utf-8")
        personas[i]["likes"]=np.sum([ i for i in personas_likes[i].values()])
        personas[i]["eval"]=len( personas_likes[i].values())
    return render_template('personas.tpl_html', personas = personas,len=len(personas))

@app.route("/like/next", methods=["POST"])
def next():
    content = request.json
    if "answer" in content:
        idPhoto =content['idPhoto']
        answer = content['answer']
        profile = content['profile']
        print(profile,idPhoto,answer)
        personas_likes[profile][idPhoto]=answer
        pickle.dump(personas_likes, open(SAVE_FOLDER+"personas_likes.pkl", "wb"))
    else:
         print("request",request)
    idPhoto= np.random.randint(len(names)-1)
    b64_string = base64.b64encode(open(names[idPhoto+1], "rb").read()).decode("utf-8")
    return jsonify({"src":"data:image/png;base64,"+b64_string,"idPhoto":idPhoto})

@app.route('/like/<int:profile>')
def like(profile):
    if personas[profile]["img"]==-1:
        file= "./static/userme.jpeg"
    else:
        file = names[personas[profile]["img"]+1]
    b64_string = base64.b64encode(open(file, "rb").read()).decode("utf-8")
    return render_template('like.tpl_html', b64_string = b64_string, company_name='TestDriven.io',profile=profile,name=personas[profile]["name"],criterias=personas[profile]["criterias"])

@app.route('/about')
def about():
    return render_template('about.tpl_html', company_name='TestDriven.io')
    #return render_template('about.html')


if __name__=="__main__":
    app.run(debug=True,host="0.0.0.0")