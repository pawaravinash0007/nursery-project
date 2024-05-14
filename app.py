import streamlit as st
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score 
#----avinash---
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay,RocCurveDisplay,PrecisionRecallDisplay
from sklearn import preprocessing 
from sklearn.metrics import precision_score, recall_score , accuracy_score 

def main():
	st.title("Model Develope to rank applications for nursery schools")
	st.sidebar.title("Model Develope to rank applications for nursery schools.")
	st.markdown("Lets Predict enrollment chances")
	st.sidebar.markdown("Lets Predict enrollment chances")

	def load_data():
		nursary_data=pd.read_csv("data_nursery.csv")
		return nursary_data
	
	st.cache(persist=True)
	def split(df):
		x_temp= df.drop(["class"],axis=1)
		x=pd.get_dummies(x_temp,drop_first=True,dtype=int)
		label_encoder = preprocessing.LabelEncoder()
		df["class"]=label_encoder.fit_transform(df["class"])
		y=df["class"]
		x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=1)
		return x_train,x_test,y_train,y_test

	df=load_data()
	x_train,x_test,y_train,y_test=split(df)
	classifier = st.sidebar.selectbox("classifier",("Logistic Regression","Decision Tree","Random Forest","KNN"))

	if classifier=="Logistic Regression":
		st.sidebar.subheader("Hyper Parameters")
		st.sidebar.write("We are not using hyper Parameters")
		if st.sidebar.button("Classify",key="classifier"):
			lr_model=LogisticRegression()
			lr_model.fit(x_train,y_train)
			y_pred=lr_model.predict(x_test)
			accuracy=accuracy_score(y_pred,y_test)
			st.write("accuracy",accuracy.round(3)*100,"%")
			st.write("Precision",precision_score(y_pred, y_test, average='weighted').round(3)*100,"%")
			st.write("Recall",recall_score(y_pred, y_test, average='weighted').round(3)*100,"%")
			st.write("Input Parameters",x_test)
			st.write("Output",y_pred)

			
	if classifier=="Decision Tree":
		st.sidebar.subheader("Hyper Parameters")
		max_depth =st.sidebar.number_input("max_depth",1,100,step=1,key="max_depth")
		min_samples_split = st.sidebar.number_input("min_samples_split",1,1000,step=1,key="min_samples_split")
		min_samples_leaf=st.sidebar.number_input("min_samples_leaf",1,100,step=1,key="min_samples_leaf")
		
		if st.sidebar.button("Classify",key="classifier"):
			dtc_model=DecisionTreeClassifier(max_depth=max_depth,min_samples_split=min_samples_split,min_samples_leaf=min_samples_leaf)
			dtc_model.fit(x_train,y_train)
			y_pred=dtc_model.predict(x_test)
			accuracy=accuracy_score(y_pred,y_test)
			st.write("accuracy",accuracy.round(3)*100,"%")
			st.write("Precision",precision_score(y_pred, y_test, average='weighted').round(3)*100,"%")
			st.write("Recall",recall_score(y_pred, y_test, average='weighted').round(3)*100,"%")
			st.write("Input Parameters",x_test)
			st.write("Output",y_pred)


	

	if classifier=="Random Forest":
		st.sidebar.subheader("Hyper Parameters")
		max_depth =st.sidebar.number_input("max_depth",1,100,step=1,key="max_depth")
		min_samples_split = st.sidebar.number_input("min_samples_split",1,1000,step=1,key="min_samples_split")
		min_samples_leaf=st.sidebar.number_input("min_samples_leaf",1,100,step=1,key="min_samples_leaf")
		n_estimators=st.sidebar.number_input("n_estimators",1,1000,key="n_estimators")
		
		if st.sidebar.button("Classify",key="classifier"):
			rfc_model=RandomForestClassifier(max_depth=max_depth,min_samples_leaf=min_samples_leaf,min_samples_split=min_samples_split,n_estimators=n_estimators)
			rfc_model.fit(x_train,y_train)
			y_pred=rfc_model.predict(x_test)
			accuracy=accuracy_score(y_pred,y_test)
			st.write("accuracy",accuracy.round(3)*100,"%")
			st.write("Precision",precision_score(y_pred, y_test, average='weighted').round(3)*100,"%")
			st.write("Recall",recall_score(y_pred, y_test, average='weighted').round(3)*100,"%")
			st.write("Input Parameters",x_test)
			st.write("Output",y_pred)



	
	if classifier=="KNN":
		st.sidebar.subheader("HyperParameters")
		n_neighbors =st.sidebar.number_input("n_neighbors",1,100,step=1,key="n_neighbors")
		if st.sidebar.button("Classify",key="classifier"):
			knn_model=RandomForestClassifier()
			knn_model.fit(x_train,y_train)
			y_pred=knn_model.predict(x_test)
			accuracy=accuracy_score(y_pred,y_test)
			st.write("accuracy",accuracy.round(3)*100,"%")
			st.write("Precision",precision_score(y_pred, y_test, average='weighted').round(3)*100,"%")
			st.write("Recall",recall_score(y_pred, y_test, average='weighted').round(3)*100,"%")
			st.write("Input Parameters",x_test)
			st.write("Output",y_pred)
        
         

	if st.sidebar.checkbox("Show raw data",False):
		st.subheader("Raw Data")
		st.write(df)



if __name__ == '__main__':
    main()
	
