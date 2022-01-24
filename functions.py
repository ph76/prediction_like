from imutils import paths
import pickle
import cv2
import os, os.path
from sklearn.cluster import DBSCAN
from imutils import build_montages
import face_recognition
import numpy as np
import pandas as pd
import random
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import matplotlib.patches as mpatches
from matplotlib.offsetbox import (DrawingArea, OffsetImage,AnnotationBbox)
import matplotlib.pyplot as plt

from PIL import Image, ImageDraw
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from matplotlib.collections import PatchCollection
from matplotlib import cm
import seaborn as sns
    

def thumb(filename,newfile):
    # load image
    img = Image.open(filename).convert('RGB')
    # crop image 
    width, height = img.size
    x = (width - height)//2
    img_cropped = img.crop((x, 0, x+height, height))
    
    # create grayscale image with white circle (255) on black background (0)
    mask = Image.new('L', img_cropped.size)
    mask_draw = ImageDraw.Draw(mask)
    width, height = img_cropped.size
    mask_draw.ellipse((0, 0, width, height), fill=255)
    #mask.show()
    # add mask as alpha channel
    img_cropped.putalpha(mask)
    # save as png which keeps alpha channel 
    img_cropped = img_cropped.resize((120,120), Image.ANTIALIAS)
    img_cropped.save(newfile)
    
    
    

def augmented_dendrogram(*args, **kwargs):

    ddata = dendrogram(*args, **kwargs)

    if not kwargs.get('no_plot', False):
        for i, d in zip(ddata['icoord'], ddata['dcoord']):
            y = 0.5 * sum(i[1:3])
            x = d[1]
            plt.plot(x, y, 'ro')
            plt.annotate("%.3g" % x, (x, y), xytext=(0, -8),
                         textcoords='offset points',
                         va='top', ha='center')
    return ddata

def get_names(leave,Z,names):
    while leave>=len(names):
        leave= Z[int(leave-len(names))][0]
    return names[int( leave-0)+1]

def plot_dendrogram(Z, names,level):
    fig=plt.figure(figsize=(14,35))
    ax = fig.add_subplot()
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('distance')
    
    ddata =dendrogram(
        Z,
         labels =names,
        orientation = "right",
        leaf_font_size=10,
        p=level,
        truncate_mode='level'
    )
    i=0
    for l in ddata['leaves']:
        arr_img = plt.imread(get_names(l,Z,names), format='png')
        imagebox = OffsetImage(arr_img, zoom=0.4)
        imagebox.image.axes = ax
        ab = AnnotationBbox(imagebox, (-15,(i+1)*1./len(ddata['leaves'])),
                    xybox=(0, -7),
                    xycoords=("data", "axes fraction"),
                    boxcoords="offset points",
                    box_alignment=(.5, 1),
                    bboxprops={"edgecolor" : "none"})
        ax.add_artist(ab)
        i+=1
    i=0
    for ic, d in zip(ddata['icoord'], ddata['dcoord']):
        y = 0.5 * sum(ic[1:3])
        x = d[1]
        arr_img = plt.imread(get_names( len(names)*2-len(ddata['leaves'])+i ,Z,names), format='png')
        imagebox = OffsetImage(arr_img, zoom=0.3)
        imagebox.image.axes = ax
        ld=len(ddata['leaves'])
        r=10*(ld)
        ab = AnnotationBbox(imagebox, (x+15,   1./ld+ (y-r*0.5/ld)/r),
                    xybox=(0, -7),
                    xycoords=("data", "axes fraction"),
                    boxcoords="offset points",
                    box_alignment=(.5, 1),
                    bboxprops={"edgecolor" : "none"})
        ax.add_artist(ab)
        i+=1 
    plt.show()
    return ddata

def distance(i,j,X_std):
    return np.sum((X_std[i]-X_std[j])*(X_std[i]-X_std[j]))

def get_nearest(i,X_std):
    j_nearest=-1
    dist=0
    for j in range(len(X_std)):
        if i==j:
            continue
        dist_cdt= distance(i,j,X_std)
        if(dist_cdt==0):
            continue
        if j_nearest==-1:
            j_nearest=j
            dist= dist_cdt
            continue
        if dist_cdt < dist:
            j_nearest=j
            dist= dist_cdt
    return (j_nearest, dist)

def show_nearest(arrayi,names,X_std):
    data=[]
    faces=[]
    title = "Simil Faces "
    for i in arrayi:
        faces.append(cv2.imread( names[i+1]))
    for i in arrayi:
        j,dist = get_nearest(i,X_std)
        faces.append(cv2.imread( names[j+1]))
    fig=plt.figure(figsize=(14,7))
    montage = build_montages(faces, (192, 192), ( len(arrayi),2))[0]
    plt.title(title)
    plt.imshow(cv2.cvtColor(montage, cv2.COLOR_BGR2RGB))
        
def pareto(data) :

    from matplotlib.ticker import PercentFormatter

    y = list(data)

    x = range(len(data))

    ycum = np.cumsum(y)/sum(y)*100

   
    fig, ax = plt.subplots(figsize=(10,5))

    ax.bar(x,y,color="yellow")

    ax2 = ax.twinx()

    ax2.plot(x,ycum, color="C1", marker="D", ms=4)

    ax2.axhline(y=80,color="r")

    ax2.yaxis.set_major_formatter(PercentFormatter())

    ax.tick_params(axis="y", colors="C0")

    ax2.tick_params(axis="y", colors="C1")

    plt.ylim(0,110)

    plt.show()

def get_nearest_from(coord,dim,box,data_sortie):
    j_nearest=-1
    dist=-1
    coord_=[6,7]
    p_=38
    for j in range(len(data_sortie)):
        coordJ =np.array( [ data_sortie[j][h] for h in dim ])
        
        coef = np.array( [    (0. if (i in dim) else 1.) for i in range(len(data_sortie[j])) ])
        p_cdt= np.sum( (coef*data_sortie[j])*(coef*data_sortie[j]))
        
        dist_cdt= np.sum((coordJ-coord)*(coordJ-coord))
        if(coordJ[0]<box[0][0] or coordJ[0]>box[0][1] or coordJ[1]<box[1][0] or coordJ[1]>box[1][1]):
            continue
        if  (dist==-1) or (p_cdt < p_):
            j_nearest=j
            dist= dist_cdt
            coord_=coordJ
            p_=p_cdt
    return (j_nearest, dist,coord_)
def get_nearest_from0(coord,data_sortie,labels=None,onlylabel=None):
    j_nearest=-1
    dist=-1
    for j in range(len(data_sortie)):
        if onlylabel is not None and labels[j]!=onlylabel:
            continue
        coordJ = data_sortie[j]
        dist_cdt= np.sum((coordJ-coord)*(coordJ-coord))
        if  (dist==-1) or (dist_cdt < dist):
            j_nearest=j
            dist= dist_cdt
            coord_=coordJ
    return (j_nearest, dist,coord_)

def get_farests_from(coord,data_sortie,limit):
    d={}
    for j in range(len(data_sortie)):
        coordJ=data_sortie[j]
        d[j]= np.sum((coordJ-coord)*(coordJ-coord))
    sd= pd.Series(d)
    sd= sd.sort_values(ascending=False)
    return sd.index[:limit]

def get_nearests_from(coord,data_sortie,limit):
    d={}
    for j in range(len(data_sortie)):
        coordJ=data_sortie[j]
        d[j]= np.sum((coordJ-coord)*(coordJ-coord))
    sd= pd.Series(d)
    sd= sd.sort_values(ascending=True)
    return sd.index[:limit+1]

def display_img(file,x,y,ax,zoom):
    arr_img = plt.imread(file, format='png')
   
    imagebox = OffsetImage(arr_img, zoom=zoom)
    imagebox.image.axes = ax
    ab = AnnotationBbox(imagebox, (x, y),
                    xybox=(0, 0),
                    xycoords=("data", "axes fraction"),
                    boxcoords="offset points",
                    box_alignment=(-0.1, -0.1),
                    bboxprops={"edgecolor" : "none"})
    ax.add_artist(ab)
def show_img(fig,file,x,y,w,h):
    im = plt.imread(file)
    newax = fig.add_axes([x,y,w,h], anchor='SW', zorder=1)
    newax.imshow(im)
    newax.axis('off')
    return newax


def show_mapface(dim,size,data_sortie,names):
    zoom= 4./size
    print(zoom)
    fig, ax = plt.subplots(figsize=(12,12))
    xmin=np.min(data_sortie[dim[0]])
    xmax=np.max(data_sortie[dim[0]])
    xstep=1.*(np.max(data_sortie[dim[0]])-np.min(data_sortie[dim[0]]))/size
    ymin=np.min(data_sortie[dim[1]])
    ymax=np.max(data_sortie[dim[1]])
    ystep=1.*(np.max(data_sortie[dim[1]])-np.min(data_sortie[dim[1]]))/size
    print(xstep,ystep)
   
    for i in range(size):
        for j in range(size):
            x= xmin + xstep*i
            y= ymin+ ystep*j
            box=[[x,x+xstep],[y,y+ystep]]
            im,dist,c= get_nearest_from(np.array([x,y]),dim,box,data_sortie)
            print(i,"x",j,"=",im," - ",dist,"\r",end="")
            if im>-1:
                #display_img(names[im+1],c[0],c[1],ax,zoom)
                display_img(names[im+1],(x-xmin)/(xmax-xmin),(y-ymin)/(ymax-ymin),ax,zoom)
    ax.set_title("Map Face axes(" +  str(dim[0]) + "," +str(dim[1]) +")",
             fontweight ="bold")           
    plt.show()
    

def nb_bycluster(ilabel,labels):
    n=0
    for i in labels:
        if labels[i]==ilabel:
            n+=1
    return n

def clustering(n_clusters,data):
    kmeans = KMeans(
  init="random",
  n_clusters=n_clusters,
   n_init=10,
 max_iter=300,
   random_state=42
 )
    kmeans.fit(data)
    return (kmeans.labels_,kmeans.cluster_centers_,kmeans)


def show_cardface(file,coords,labels,centers,data_sortie,kms,names):
    labeled  =kms.predict([coords])
    only=labeled[0]
    height=1
    size=12
    fig, ax = plt.subplots(figsize=(12,height*12/(size)))
    xmin=0
    xmax=size
    xstep=1./size
    ymin=0
    ymax=height
    ystep=1./height
    
    
    if height<size:
        w=0.6/height
        h=0.6/height
        mutation_aspect=1./height
    patches = []
    colors=cm.rainbow(np.linspace(0,1,len(centers)))
    
    for i0 in range(height):
        i=only
        ih=0
        
        ax0 = fig.add_axes([0.5/size,(ih+0.05)*ystep,1,ystep*0.95],anchor='SW',  zorder=0)#,facecolor="red")#,
        
        ax0.axis('off')
        #ax0.set_title( "Groupe "+ str(i) + "  ("+str(labels.tolist().count(i))+")")
       
        im0=0
        color=colors[i]
        #(np.random.random(),np.random.random(),np.random.random())
        
        ax0.add_patch(mpatches.Ellipse([ 0.5/size,0.5],width= 1./size,height=1 ,fill=True,ec="none",color=color))
        ax0.add_patch(mpatches.Ellipse([1- 0.5/size,0.5],width= 1./size,height=1 ,fill=True,ec="none",color=color))
        ax0.add_patch(mpatches.Rectangle([ 0.5/size,0],width= 1-1./size,height=1 ,fill=True,ec="none",color=color))
        ax0.add_patch(mpatches.Rectangle([ 0.585,0.1],width= 0.003,height=0.8 ,fill=True,ec="none",color="white"))
        ax0.text(0.13,0.8, " Groupe "+ str(i) + "/"+ str(len(centers))+"  ("+str(labels.tolist().count(i))+")", ha="center", family='sans-serif', color="white",size=12,weight="bold")
        ax0.text(0.7,0.8,  "Les plus éloignés", ha="center", family='sans-serif', color="white",size=12,weight="bold")
        
        
        for si in range(4):
            emin=dmin(data_sortie,si)
            emax=dmax(data_sortie,si)
            ax0.text(0.08,0.75-0.15*(si+1),  str(si), ha="center", family='sans-serif', color="white",size=12,weight="bold")
            ax0.add_patch(mpatches.Rectangle([ 0.09,0.75-0.15*(si+1)],width= 0.08,height=0.1  ,fill=True,ec="gray",color=color,alpha=0.3))
            ax0.add_patch(mpatches.Rectangle([ 0.09,0.75-0.15*(si+1)],width= 0.08*(coords[si]-emin)/(emax-emin),height=0.1  ,fill=True,ec="none",color="white"))
           
            
        ax_im=show_img(fig,file,0.7/size,(ih+0.22)*ystep,w,h)
        ax0.add_patch(mpatches.Ellipse([ 0.5/size,0.5],width=  0.7/size,height=0.7,fill=True, lw=0.3,ec="white",color="white"))
        
        nearests=get_nearests_from(coords,data_sortie,5)
        farests=get_farests_from(coords,data_sortie,5)
        for j in range((size-2)):
            x= (xmin +(2.7+j)* xstep)
            if j< (size-2)//2:
                show_img(fig,names[nearests[j]+1],x,(ih+0.2)*ystep,w,h)
            else:
                show_img(fig,names[farests[j- (size-2)//2]+1],x,(ih+0.2)*ystep,w*0.8,h*0.8)
    ax.axis('off')
    #collection = PatchCollection(patches, cmap=plt.cm.hsv, alpha=1)
    #colors = np.linspace(0, 1, len(patches))
    #collection.set_array(np.array(colors))
    #ax.add_collection(collection)
    plt.show()
    

def show_clusterfaces(labels,centers,data_sortie,names,only=None):
    if only is None:
        height=len(centers)
    else:
        height=1
    
    size=12
    fig, ax = plt.subplots(figsize=(12,height*12/(size)))
    xmin=0
    xmax=size
    xstep=1./size
    ymin=0
    ymax=height
    ystep=1./height
    
    
    if height<size:
        w=0.6/height
        h=0.6/height
        mutation_aspect=1./height
    else:
        w=0.6/size
        h=0.6/size
        mutation_aspect=1./size
    patches = []
    colors=cm.rainbow(np.linspace(0,1,len(centers)))
    
    for i0 in range(height):
        if only is None:
            i=i0
            ih=i0
        else:
            i=only
            ih=0
        im,dist,c= get_nearest_from0(np.array(centers[i]),data_sortie,labels=labels,onlylabel=i)
        #display_img(names[im+1],0.5/size,i/ymax,ax,zoom)
        
        ax0 = fig.add_axes([0.5/size,(ih+0.05)*ystep,1,ystep*0.95],anchor='SW',  zorder=0)#,facecolor="red")#,
        
        ax0.axis('off')
        #ax0.set_title( "Groupe "+ str(i) + "  ("+str(labels.tolist().count(i))+")")
       
        im0=0
        color=colors[i]
        #(np.random.random(),np.random.random(),np.random.random())
        
        ax0.add_patch(mpatches.Ellipse([ 0.5/size,0.5],width= 1./size,height=1 ,fill=True,ec="none",color=color))
        ax0.add_patch(mpatches.Ellipse([1- 0.5/size,0.5],width= 1./size,height=1 ,fill=True,ec="none",color=color))
        ax0.add_patch(mpatches.Rectangle([ 0.5/size,0],width= 1-1./size,height=1 ,fill=True,ec="none",color=color))
        ax0.add_patch(mpatches.Rectangle([ 0.585,0.1],width= 0.003,height=0.8 ,fill=True,ec="none",color="white"))
        ax0.text(0.13,0.8,  "Groupe "+ str(i) + "/"+ str(len(centers))+"  ("+str(labels.tolist().count(i))+")", ha="center", family='sans-serif', color="white",size=12,weight="bold")
        ax0.text(0.7,0.8,  "Les plus éloignés", ha="center", family='sans-serif', color="white",size=12,weight="bold")
        
        
        for si in range(4):
            emin=dmin(data_sortie,si)
            emax=dmax(data_sortie,si)
            ax0.text(0.08,0.75-0.15*(si+1),  str(si), ha="center", family='sans-serif', color="white",size=12,weight="bold")
            ax0.add_patch(mpatches.Rectangle([ 0.09,0.75-0.15*(si+1)],width= 0.08,height=0.1  ,fill=True,ec="gray",color=color,alpha=0.3))
            ax0.add_patch(mpatches.Rectangle([ 0.09,0.75-0.15*(si+1)],width= 0.08*(centers[i][si]-emin)/(emax-emin),height=0.1  ,fill=True,ec="none",color="white"))
           
            
        ax_im=show_img(fig,names[im+1],0.7/size,(ih+0.22)*ystep,w,h)
        ax0.add_patch(mpatches.Ellipse([ 0.5/size,0.5],width=  0.7/size,height=0.7,fill=True, lw=0.3,ec="white",color="white"))
        
        farests=get_farests_from(centers[i],data_sortie,5)
        for j in range((size-2)):
            x= (xmin +(2.7+j)* xstep)
            if j< (size-2)//2:
                while im0<len(labels) and labels[im0]!=i :
                    im0+=1
                #display_img(names[im+1],c[0],c[1],ax,zoom)
                if( im0<len(labels)):
                    show_img(fig,names[im0+1],x,(ih+0.2)*ystep,w,h)
                im0+=1
            else:
                 show_img(fig,names[farests[j- (size-2)//2]+1],x,(ih+0.2)*ystep,w*0.8,h*0.8)
    ax.axis('off')
    #collection = PatchCollection(patches, cmap=plt.cm.hsv, alpha=1)
    #colors = np.linspace(0, 1, len(patches))
    #collection.set_array(np.array(colors))
    #ax.add_collection(collection)
    plt.show()

def dmin(data,i):
    x=[]
    for v in data:
        x.append(v[i])
    return np.min(x)
def dmax(data,i):
    x=[]
    for v in data:
        x.append(v[i])
    return np.max(x)

def ddim(data,i,labels,ilabel):
    x=[]
    for idx,v in enumerate(data):
        if(labels[idx]==ilabel):
            x.append(v[i])
    return x

def show_mapcentroids(title,dim,data_sortie,labels,centers,names):
    fig, ax = plt.subplots(figsize=(12,12))

    xmin=dmin(data_sortie,dim[0])
    xmax=dmax(data_sortie,dim[0])
    ymin=dmin(data_sortie,dim[1])
    ymax=dmax(data_sortie,dim[1])
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    colors=cm.rainbow(np.linspace(0,1,len(centers)))
    #plt.scatter(ddim( data_sortie,dim[0]),ddim(data_sortie,dim[1]), s=1, c=colors,cmap=colors, alpha=0.5)
    ax.text(xmin,ymin,  "ACP("+str(dim[0])+","+str(dim[1])+") "+ str(len(centers))+ " Groupes. " , ha="center", family='sans-serif', color="gray",size=20,weight="bold")
    ax.axis('off')
    for i,center in enumerate(centers):
        xs=ddim( data_sortie,dim[0],labels,i)
        ys= ddim(data_sortie,dim[1],labels,i)
        sns.kdeplot( x=xs, y=ys,color=colors[i], shade=True,alpha=0.15)
        ax.scatter(xs,
                    ddim(data_sortie,dim[1],labels,i),c=[colors[i]],s=20,  alpha=0.3)
        ax.scatter(centers[i][dim[0]],
                   centers[i][dim[1]],c=[colors[i]],s=360,  alpha=0.5)
        
    for i,center in enumerate(centers):
        x= (center[dim[0]]-xmin)/(xmax-xmin)
        y= (center[dim[1]]-ymin)/(ymax-ymin)
        
        #x= center[dim[0]]
        #y= center[dim[1]]
        size=0.05
        im,dist,c= get_nearest_from0(np.array(center),data_sortie,labels=labels,onlylabel=i)
        #ax.scatter((ddim( data_sortie,dim[0],labels,i)-xmin)/(xmax-xmin),
        #            (ddim(data_sortie,dim[1],labels,i)-ymin)/(ymax-ymin),c=[colors[i]],s=20,  alpha=0.5)
        ax0 = fig.add_axes([x,y,size,size],anchor='SW',  zorder=1)#,facecolor="red")#,
        ax0.axis('off')
        ax0.text(0,0,  "G. "+ str(i) , ha="center", family='sans-serif', color=colors[i],size=10,weight="bold")
        ax0.add_patch(mpatches.Ellipse([ 0.5,0.5],width= 1,height=1 ,fill=True,ec="none",color=colors[i]))
        ax0.add_patch(mpatches.Ellipse([0.5,0.5],width= 0.9,height=0.9 ,fill=True,ec="none",color="white"))
        show_img(fig,names[im+1],x+size*0.1,y+size*0.1,size*0.8,size*0.8)
    ax.set_title(title,  fontweight ="bold")
        #ax0.add_patch(mpatches.Ellipse([ 0.5,0.5],width= 1,height=1 ,fill=True,ec="none",color=colors[i]))
    plt.show()
    
def image_vectorize(imagePath,std_scale,pca):
    image = cv2.imread(imagePath)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # detect the (x, y)-coordinates of the bounding boxes
    # corresponding to each face in the input image
    boxes = face_recognition.face_locations(rgb,model='hog')  # or cnn
    # compute the facial embedding for the face
    if len(boxes)==0:
        print("ERROR no face detected!!")
        return
    encodings = face_recognition.face_encodings(rgb, boxes)
    # build a dictionary of the image path, bounding box location,
    # and facial encodings for the current image
    x_std= std_scale.transform(encodings)
    vectors= pca.transform(x_std)
    thumb_file="./results/thumb_.png" 
    thumb(imagePath,thumb_file)
    return (vectors[0],thumb_file)

def show_persona(persona,likes,names,labels):
    limit=10
    only=1
    height=1
    size=12
    fig, ax = plt.subplots(figsize=(12,height*12/(size)))
    xmin=0
    xmax=size
    xstep=1./size
    ymin=0
    ymax=height
    ystep=1./height
    colors=cm.rainbow(np.linspace(0,1,len(np.unique(labels))))
    
    if height<size:
        w=0.6/height
        h=0.6/height
        mutation_aspect=1./height
    patches = []
   
    
    for i0 in range(height):
        i=only
        ih=0
        
        ax0 = fig.add_axes([0.5/size,(ih+0.05)*ystep,1,ystep*0.95],anchor='SW',  zorder=0)#,facecolor="red")#,
        
        ax0.axis('off')
        #ax0.set_title( "Groupe "+ str(i) + "  ("+str(labels.tolist().count(i))+")")
       
        im0=0
        #(np.random.random(),np.random.random(),np.random.random())
        
        #+ persona["criterias"]
        ax0.text(0.09,0.8,persona["name"], ha="left", family='sans-serif', color="BLACK",size=12,weight="bold")
        ax0.text(0.09,0.5,str(persona["age"]) + " ans", ha="left", family='sans-serif', color="gray",size=12,weight="regular")
        ax0.text(0.09,0.2,"Likes", ha="left", family='sans-serif', color="gray",size=12,weight="regular")
        ax_im=show_img(fig,names[persona["img"]+1 ],0.7/size,(ih+0.1)*ystep,w*1.3,h*1.3)
  
        
        j=0
        for k in likes:
            if likes[k]:
                x= (xmin +(2.7+j)* xstep)
                ax0.add_patch(mpatches.Ellipse([ x-0.2*xstep ,0.5],width= xstep*0.8,height=0.95,fill=True, lw=0.3,ec="white",color=colors[ labels[k]]))
                show_img(fig,names[k+1],x,(ih+0.11)*ystep,w,h)
                ax0.text(x-0.18*xstep,0.8,"lbl "+ str( labels[k]), ha="center", family='sans-serif', color="white",size=8,weight="bold")
                j+=1
                if j==limit:
                    break
    ax.axis('off')
    #collection = PatchCollection(patches, cmap=plt.cm.hsv, alpha=1)
    #colors = np.linspace(0, 1, len(patches))
    #collection.set_array(np.array(colors))
    #ax.add_collection(collection)
    plt.show()
    
    
def show_predict_like(persona,title,faces,likes_perc,names,labels):
    only=1
    height=1
    size=12
    fig, ax = plt.subplots(figsize=(12,height*12/(size)))
    xmin=0
    xmax=size
    xstep=1./size
    ymin=0
    ymax=height
    ystep=1./height
    limit=4
    colors=cm.rainbow(np.linspace(0,1,len(np.unique(labels))))
    if height<size:
        w=0.6/height
        h=0.6/height
        mutation_aspect=1./height
    patches = []
   
    
    for i0 in range(height):
        i=only
        ih=0
        
        ax0 = fig.add_axes([0.5/size,(ih+0.05)*ystep,1,ystep*0.95],anchor='SW',  zorder=0)#,facecolor="red")#,
        
        ax0.axis('off')
        #ax0.set_title( "Groupe "+ str(i) + "  ("+str(labels.tolist().count(i))+")")
       
        im0=0
        #(np.random.random(),np.random.random(),np.random.random())
        
     
        ax0.text(0.09,0.8,persona["name"], ha="left", family='sans-serif', color="BLACK",size=12,weight="bold")
        ax0.text(0.09,0.5,"Prédiction", ha="left", family='sans-serif', color="gray",size=12,weight="regular")
        ax0.text(0.09,0.2,title, ha="left", family='sans-serif', color="gray",size=12,weight="regular")
        ax_im=show_img(fig,names[persona["img"]+1 ],0.7/size,(ih+0.1)*ystep,w*1.3,h*1.3)
   
        
        j=0
        for j,i in enumerate(faces):
            x= ((2.7+j)* xstep)
            ax0.add_patch(mpatches.Ellipse([ x-0.2*xstep ,0.5],width= xstep*0.8,height=0.95,fill=True, lw=0.3,ec="white",color=colors[ labels[i]]))
            show_img(fig,names[i+1],x,(ih+0.11)*ystep,w,h)
            ax0.text(x-0.18*xstep,0.8,str(likes_perc[j])+" %", ha="center", family='sans-serif', color="white",size=8,weight="bold")
            if j==limit:
                break
       
    ax.axis('off')
    #collection = PatchCollection(patches, cmap=plt.cm.hsv, alpha=1)
    #colors = np.linspace(0, 1, len(patches))
    #collection.set_array(np.array(colors))
    #ax.add_collection(collection)
    plt.show()