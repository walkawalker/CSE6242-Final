# -*- coding: utf-8 -*-
"""
Created on Mon Apr 18 14:06:28 2022

@author: whill
"""

import tkinter as tk
from tkinter import filedialog
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.cluster import KMeans, DBSCAN
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import plotly.express as px

root = tk.Tk()
root.title('Wind Power Clustering Tool')
canvas1 = tk.Canvas(root, width = 450, height = 450,  relief = 'raised')
canvas1.pack()

label1 = tk.Label(root, text='k-Means Clustering')
label1.config(font=('helvetica', 14))
canvas1.create_window(225, 70, window=label1)

label2 = tk.Label(root, text='Type Number of Clusters:')
label2.config(font=('helvetica', 8))
canvas1.create_window(112.5, 100, window=label2)

entry1 = tk.Entry(root) 
canvas1.create_window(112.5, 120, window=entry1)

label3 = tk.Label(root, text='Provide value k for Range (3,k):')
label3.config(font=('helvetica', 8))
canvas1.create_window(337.5, 100, window=label3)

entry2 = tk.Entry(root) 
canvas1.create_window(337.5, 120, window=entry2)

label5 = tk.Label(root, text='DBscan Clustering')
label5.config(font=('helvetica', 14))
canvas1.create_window(225, 190, window=label5)

label6 = tk.Label(root, text='Provide Min Samples:')
label6.config(font=('helvetica', 8))
canvas1.create_window(112.5, 215, window=label6)

entry3 = tk.Entry(root) 
canvas1.create_window(112.5, 240, window=entry3)

label7 = tk.Label(root, text='Provide Neighbors Max Distance:')
label7.config(font=('helvetica', 8))
canvas1.create_window(337.5, 215, window=label7)

entry4 = tk.Entry(root) 
canvas1.create_window(337.5, 240, window=entry4)

label9 = tk.Label(root, text='GMM Clustering')
label9.config(font=('helvetica', 14))
canvas1.create_window(225, 310, window=label9)

label10 = tk.Label(root, text='Provide Components Number:')
label10.config(font=('helvetica', 8))
canvas1.create_window(225, 340, window=label10)

entry5 = tk.Entry(root) 
canvas1.create_window(225, 360, window=entry5)

def Load_Excel_File():
    
    global df
    global dbscandf
    global gmmdf
    global resetdf
    import_file_path = filedialog.askopenfilename()
    read_file = pd.read_csv(import_file_path)
    df = pd.DataFrame(read_file,columns=['latitude','longitude']) 
    dbscandf = df.copy()
    gmmdf = df.copy()
    resetdf= df.copy()
    
browseButtonExcel = tk.Button(text=" Import Excel File ", command=Load_Excel_File, bg='green', fg='white', font=('helvetica', 10, 'bold'))
canvas1.create_window(225, 25, window=browseButtonExcel)

def resetdatakmeans():
    df = resetdf.copy()
def resetdatadb():
    dbscandf = resetdf.copy()
def resetdatagmm():
    gmmdf = resetdf.copy()

def myDBSCAN(df, min_points, epsilon):
    model = DBSCAN(epsilon, min_points).fit(df)
    labels = model.labels_
    dbscandf['label'] = labels
    return model, dbscandf

def latlong_to_miles(latlong, is_lat=True):
    if is_lat:
        return latlong * 364000 / 5280
    else:
        return latlong * 288200 / 5280

def miles_to_latlong(miles, is_lat=True):
    if is_lat:
        return miles * 5280 / 364000
    else:
        return miles * 5280 / 288200

def get_stats(model, df_w_labels):
    labels = model.labels_
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_outliers = list(labels).count(-1)
    n_inliers = len(list(labels)) - n_outliers
    X_array = pd.DataFrame(df_w_labels, columns=['latitude','longitude'])
    labels_array = pd.DataFrame(df_w_labels, columns=['label'])
    silhouette_score = metrics.silhouette_score(X_array, labels_array)

    return n_clusters, n_outliers, n_inliers, silhouette_score

def getKMeans():
    resetdatakmeans()
    root3 = tk.Tk()
    canvas3 = tk.Canvas(root3, width = 400, height = 300,  relief = 'raised')
    canvas3.pack()
    root4 = tk.Tk()
    canvas4 = tk.Canvas(root4, width = 400, height = 300,  relief = 'raised')
    canvas4.pack()
    global numberOfClusters
    global kdf1
    numberOfClusters = int(entry1.get())
    kmeans = KMeans(n_clusters=numberOfClusters, random_state = 0).fit(df)
    centroids = kmeans.cluster_centers_
    predictions = kmeans.predict(df)
    kclust = pd.DataFrame(predictions, columns=['k_cluster'])
    kvalues = kclust['k_cluster'].value_counts().sort_index()
    kcount = np.arange(0, len(centroids),1)
    kdf1 = pd.DataFrame(centroids, columns= ['Centroid_Lat', 'Centroid_Lon'])
    kdf1.insert(loc=0, column= 'k_cluster', value=kcount)
    kdf1.insert(loc=1, column= 'count_in_cluster', value=kvalues)
    label4 = tk.Label(root4, text= f'Centroid Latitudes:\n {centroids[:,0]}')
    label11 = tk.Label(root4, text= f'Centroid Longitudes:\n {centroids[:,1]}')
    canvas4.create_window(200, 50, window=label4)
    canvas4.create_window(200, 250, window=label11)
    df['k_cluster'] = kmeans.predict(df)
    figure1 = plt.Figure(figsize=(16,8), dpi=100)
    ax1 = figure1.add_subplot()
    ax1.scatter(df['latitude'], df['longitude'], c= kmeans.labels_.astype(float), s=50, alpha=0.5)
    ax1.scatter(centroids[:, 0], centroids[:, 1], c='red', s=50)
    scatter1 = FigureCanvasTkAgg(figure1, root3) 
    scatter1.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH)
    
def Kdataframe(df):
    numberOfClusters = int(entry1.get())
    kmeans = KMeans(n_clusters=numberOfClusters, random_state = 0).fit(df)
    centroids = kmeans.cluster_centers_
    predictions = kmeans.predict(df)
    kclust = pd.DataFrame(predictions, columns=['k_cluster'])
    kvalues = kclust['k_cluster'].value_counts().sort_index()
    kcount = np.arange(0, len(centroids),1)
    centroid_kclust = pd.DataFrame(centroids, columns= ['Centroid_Lat', 'Centroid_Lon'])
    centroid_kclust.insert(loc=0, column= 'k_cluster', value=kcount)
    centroid_kclust.insert(loc=1, column= 'count_in_cluster', value=kvalues)
    return centroid_kclust

    
def calculate_WSS(points, kmax):
    sse = []
    for k in range(3, kmax+1):
        kmeans = KMeans(n_clusters = k).fit(points)
        centroids = kmeans.cluster_centers_
        pred_clusters = kmeans.predict(points)
        curr_sse = 0

        for i in range(len(points)):
            curr_center = centroids[pred_clusters[i]]
            curr_sse += (points[i, 0] - curr_center[0]) ** 2 + (points[i, 1] - curr_center[1]) ** 2

        sse.append(curr_sse)
    return sse, np.arange(3,kmax+1,1) 
    sse = []
    for k in range(3, kmax+1):
        kmeans = KMeans(n_clusters = k).fit(points)
        centroids = kmeans.cluster_centers_
        pred_clusters = kmeans.predict(points)
        curr_sse = 0

# calculate square of Euclidean distance of each point from its cluster center and add to current WSS
        for i in range(len(points)):
            curr_center = centroids[pred_clusters[i]]
            curr_sse += (points[i, 0] - curr_center[0]) ** 2 + (points[i, 1] - curr_center[1]) ** 2

        sse.append(curr_sse)
    return sse, np.arange(3,kmax+1,1) 

def plot_WSS():
    root2 = tk.Tk()
    canvas2 = tk.Canvas(root2, width = 400, height = 300,  relief = 'raised')
    canvas2.pack()
    wss, krange = calculate_WSS(np.array(df),int(entry2.get()))
    figure2 = plt.Figure(figsize=(16,8), dpi=100)
    ax2 = figure2.add_subplot()
    ax2.plot(krange, wss)
    ax2.set_title("WSS vs K")
    ax2.set_xticks(krange)
    ax2.set_ylabel("WSS")
    ax2.set_xlabel("K")
    plot1 =FigureCanvasTkAgg(figure2, root2)
    plot1.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH)

def getDBscanMod():
    resetdatadb()
    min_points = int(entry3.get()) # Going to be chosen by user
    epsilon_in_miles = int(entry4.get())
    epsilon_in_latlong = miles_to_latlong(epsilon_in_miles)
    model, df_w_labels = myDBSCAN(dbscandf, min_points, epsilon_in_latlong)
    n_clusters, n_outliers, n_inliers, silhouette_score = get_stats(model, df_w_labels)
    root5 = tk.Tk()
    canvas5 = tk.Canvas(root5, width = 400, height = 300,  relief = 'raised')
    canvas5.pack()
    label8 = tk.Label(root5, text= f'Clusters: {n_clusters}\n Outliers: {n_outliers}\n Inliers: {n_inliers}\n Silhouette Score: {silhouette_score}')
    canvas5.create_window(200, 50, window=label8)
    
def myGMM(df, n_clusters):
    gm = GaussianMixture(n_components=n_clusters, random_state=0).fit(df)
    gmmlabels = gm.predict(df)
    return gmmlabels
def plotGMMSilhouette(df):
    root7 = tk.Tk()
    canvas7 = tk.Canvas(root7, width = 200, height = 300,  relief = 'raised')
    canvas7.pack()
    X_array = pd.DataFrame(df, columns=['latitude','longitude'])
    labels_array = pd.DataFrame(df, columns=['labels'])
    figure4 = plt.Figure(figsize=(16,8), dpi=100)
    ax1 = figure4.add_subplot()
    ax1.set_xlim([-0.1, 1])
    ax1.set_ylim([0, len(X_array) + (labels_array['labels'].max() + 1) * 10])

    silhouette_avg = metrics.silhouette_score(X_array, labels_array)

    sample_silhouette_values = metrics.silhouette_samples(X_array, labels_array)

    y_lower = 10
    for i in range(0, labels_array['labels'].max()+1):

        ith_cluster_silhouette_values = sample_silhouette_values[labels_array['labels'] == i]
        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i
        color = cm.nipy_spectral(float(i) / (labels_array['labels'].max()+1))
        ax1.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            ith_cluster_silhouette_values,
            facecolor=color,
            edgecolor=color,
            alpha=0.7)
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        y_lower = y_upper + 10  

    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")
    ax1.text(.75, y_upper*0.5, f"Average Silhouette Score\n {silhouette_avg}", fontsize=15)
    ax1.set_yticks([])
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
    plot4 = FigureCanvasTkAgg(figure4, root7)
    plot4.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH)
    
    
def runGMM():
    resetdatagmm()
    gmmdf['labels']  = myGMM(gmmdf, int(entry5.get()))
    root6 = tk.Tk()
    canvas6 = tk.Canvas(root6, width = 200, height = 300,  relief = 'raised')
    canvas6.pack()
    plotGMMSilhouette(gmmdf)
    figure3 = plt.Figure(figsize=(16,8), dpi=100)
    ax3 = figure3.add_subplot()
    ax3.scatter(gmmdf['longitude'], gmmdf['latitude'], c=gmmdf['labels'], s=40, cmap='viridis')
    ax3.set_title("GMM Clustering")
    plot3 = FigureCanvasTkAgg(figure3, root6)
    plot3.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH)

def writetocsv():
    kdf2 = df.merge(kdf1, how='left',on= 'k_cluster')
    kdf1.to_csv('k_means_centroids.csv')
    kdf2.to_csv('k_means_full.csv')
    dbscandf.to_csv('dbscan_data.csv')
    gmmdf.to_csv('gmm_data.csv')
    #root.quit()


mapButton = tk.Button(text=' Display Clusters ', command=getKMeans, bg='brown', fg='white', font=('helvetica', 10, 'bold'))
canvas1.create_window(112.5, 150, window=mapButton)
graphButton = tk.Button(text=' Elbow Graph ', command=plot_WSS, bg='brown', fg='white', font=('helvetica', 10, 'bold'))
canvas1.create_window(337.5, 150, window=graphButton)
dbButton = tk.Button(text=' Run DBScan ', command=getDBscanMod, bg='brown', fg='white', font=('helvetica', 10, 'bold'))
canvas1.create_window(225, 270, window=dbButton)
gmmButton = tk.Button(text=' Run GMM ', command=runGMM, bg='brown', fg='white', font=('helvetica', 10, 'bold'))
canvas1.create_window(225, 390, window=gmmButton)
writeButton = tk.Button(text=' Save Data ', command=writetocsv, bg='blue', fg='white', font=('helvetica', 10, 'bold'))
canvas1.create_window(225, 420, window=writeButton)
root.mainloop()
