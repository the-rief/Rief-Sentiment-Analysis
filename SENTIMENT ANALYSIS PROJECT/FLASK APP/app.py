# Library imports
import nltk
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import seaborn as sns
import pandas as pd
import snscrape.modules.twitter as sntwitter
import numpy as np
import spacy
import re
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import requests
import io
import string
from spacy.lang.en.stop_words import STOP_WORDS
from flask import Flask, request, jsonify, render_template, session, redirect, url_for, send_file
from flask_wtf import FlaskForm
from wtforms import FileField, SubmitField
from werkzeug.utils import secure_filename
from wtforms.validators import InputRequired
import os
import nltk
import mysql
import mysql.connector as sql

nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

all_stopwords = stopwords.words('english')
all_stopwords.remove('not')



# Create the app instance
app = Flask(__name__)

#configure folder for files uploaded
app.config['UPLOAD_FOLDER'] = 'static/files'

#functions for menu
#home menu
@app.route("/homemenu")
def homemenu():
    return render_template("home.html")

#sentiment menu
@app.route("/sentimentmenu")
def sentimentmenu():
    return render_template("sentiment.html")

#uploadmenu menu
@app.route("/analysismenu")
def analysismenu():
    return render_template("analysis.html")

#tweets menu
@app.route("/tweetsmenu")
def tweetsmenu():
    return render_template("gettweets.html")

#logout menu
@app.route("/logoutmenu")
def logoutmenu():
    return render_template("index.html")

#download button
@app.route("/download")
def download():
    return render_template("gettweets.html")

#database
#connecting to the database
connection = sql.connect(host='localhost', port='3306', database='sentimentdb', user='root', password='')
cursor = connection.cursor()

#set secret key for session
app.secret_key = "super secret key"

#adding a route for home page
@app.route("/home")
def home():
    return render_template('home.html', username= session['username'])


# login 
@app.route('/login', methods=['GET', 'POST'])
def login():
    msg=''
    if request.method=='POST':
        username = request.form['username']
        password = request.form['password']
        s_query = 'SELECT * FROM adminlogin WHERE username=%s AND password=%s'
        cursor.execute(s_query, (username, password,))
        record = cursor.fetchone()
        if record:
            session['loggedin']=True
            session['username']=record[1]
            return redirect(url_for('home'))
        else:
            msg='incorrect username/password. Try Again!'
    return render_template('index.html', msg=msg)    


#to the registeration signing in page
@app.route("/registering")
def registering():
    return render_template('registration.html')

@app.route('/signin',methods=['POST'])
def signin():
    if request.method == 'POST':
        return render_template('registration.html')
    else:
        return redirect(url_for("registering"))

#the button to the sentiments page    
@app.route('/sentiment',methods=['POST'])
def sentiment():
    if request.method == 'POST':
        return render_template('sentiment.html')
    else:
        return redirect(url_for("home"))

#the button to the upload page
@app.route('/upload',methods=['POST'])
def upload():
    if request.method == 'POST':
        #form = UploadFileForm()
        #if form.validate_on_submit():
           # file = form.file.data #first grab the file
           # file.save(os.path.join(os.path.abspath(os.path.dirname(__file__)), app.config['UPLOAD_FOLDER'], secure_filename(file.filename))) #saves file
            #trying to read the file
           # dataset=(file.filename)
          #  c=pd.read_csv(io.StringIO(dataset))
            #dataset = pd.read_csv(f , delim_whitespace=True, quoting = 3)
            
            
            #return render_template('upload.html', form=form, upload_text='File has been uploaded')
        
        return render_template('gettweets.html')
    else:
        return redirect(url_for("home"))

#the upload page
class UploadFileForm(FlaskForm):
    file = FileField("File", validators = [InputRequired()])
    submit = SubmitField("Upload File")
 


# Load trained Classifier
model = joblib.load('SVCmodel_classifier.pkl')
vec = joblib.load('SVCmodel_vectorizer.pkl')

stopwords = list(STOP_WORDS)


# calling the route index
@app.route('/')
def sent():
    return render_template('index.html')

#the predictor function
@app.route('/predict',methods=['POST'])
def predict():
    new_review = [str(x) for x in request.form.values()]
#     data = pd.DataFrame(new_review)
#     data.columns = ['new_review']
    new_review = vec.transform(new_review)
    predictions = model.predict(new_review)
    if predictions==0:
        return render_template('sentiment.html', prediction_text='Negative 😞😔')
    else:
        return render_template('sentiment.html', prediction_text='Positive 🥰🥰')


#tweets for 2022 first 6 months
@app.route('/tweets',methods=['POST'])
def tweets():
    #return render_template('gettweets.html', prediction_text='Negative 😞😔')
    query = [str(x) for x in request.form.values()]
    #query = " uber_kenya until:2022-06-01 since:2022-01-01"
    tweets = []
    limit = 100

    for tweet in sntwitter.TwitterSearchScraper(str(query)).get_items():
        if len(tweets) == limit:
            break
        else:
            tweets.append([tweet.date, tweet.user.username, tweet.content])

    df = pd.DataFrame(tweets, columns=['Date', 'User', 'Tweet'])
    #creating a csv file for the data
    df.to_csv('uber_tweets.csv')
    #try the auto model
    dataset = pd.read_csv('uber_tweets.csv')
    #dataset.head()
    
    
    return render_template('gettweets.html', tweets2_text="Tweets Have been generated successfully")

@app.route('/gotosentimentpage',methods=['POST'])
def gotosentimentpage():
    return render_template('sentimentpage.html')


#importing countvectorizer

vec2 = joblib.load('Uber_BoW_Sentiment_Model.pkl')
model2 = joblib.load('UberLogisticmodel_classifier.pkl')

#sentiment analysis for csv
@app.route('/sentimentpage',methods=['POST'])
def sentimentpage():
    #with open('uber_tweets.csv') as f:
    #lines = f.readlines()
    dataset = pd.read_csv('uber_tweets.csv')
    dataset.columns = ['id', 'date', 'user', 'tweet']
    #we drop the unnecessary columns
    # dataset.drop(['id', 'date', 'user'], axis=1)
    dataset.drop(dataset.columns[[0, 1, 2]], axis=1, inplace=True)
    corpus=[]

    for i in range(0, 100):
        review = re.sub('[^a-zA-Z]', ' ', dataset['tweet'][i])
        # Remove mentions
        review = re.sub(r'@[A-Za-z0-9_]+', '', review)
        # Remove hashtags
        review = re.sub(r'#[A-Z0-9]+', '', review)
        # Remove retweets:
        review = re.sub(r'RT : ', '', review)
        # Remove urls
        review = re.sub(r'https?:\/\/[A-Za-z0-9\.\/]+', '', review)
        #remove amp
        review = re.sub(r'&amp;', '', review)
        #rempve strange characters
        review = re.sub(r'ðŸ™', '', review)
        #remove new lines
        review = re.sub(r'\n', ' ', review)
        review = review.lower()
        review = review.split()
        review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
        review = ' '.join(review)
        corpus.append(review)
    #transforming the corpus
    X_fresh = vec2.transform(corpus).toarray()
    #fitting the model
    #doing the prediction
    y_pred = model2.predict(X_fresh)
    #printing the new labelled dataset
    dataset['Uber_predicted_label'] = y_pred.tolist() #tweets3_text=(dataset.tail())
    dataset.to_csv('predicted_ubertweets.csv')
    
    
    return render_template('table.html', tables=[dataset.to_html()], titles=['predicted'])
    
#business insights
#the button to the table page    
@app.route('/tablechart',methods=['POST'])
def tablechart():
    if request.method == 'POST':
        dataset = pd.read_csv('predicted_ubertweets.csv')
        return render_template('table.html', tables=[dataset.to_html()], titles=['predicted'])
    else:
        return redirect(url_for("home"))

#the button to the negative tweets table page    
@app.route('/negativetweets',methods=['POST'])
def negativetweets():
    if request.method == 'POST':
        dataset = pd.read_csv('predicted_ubertweets.csv')
        tweet_negative = dataset[dataset["Uber_predicted_label"] == 0]
        tweet_positive = dataset[dataset["Uber_predicted_label"] == 1]
        return render_template('negativetweets.html', tables=[tweet_negative.to_html()], titles=['predicted'])
    else:
        return redirect(url_for("home"))

#the button to the negative tweets table page    
@app.route('/positivetweets',methods=['POST'])
def positivetweets():
    if request.method == 'POST':
        dataset = pd.read_csv('predicted_ubertweets.csv')
        tweet_negative = dataset[dataset["Uber_predicted_label"] == 0]
        tweet_positive = dataset[dataset["Uber_predicted_label"] == 1]
        return render_template('positivetable.html', tables=[tweet_positive.to_html()], titles=['predicted'])
    else:
        return redirect(url_for("home"))
    
        
#function for percentage
def calc_percentage(x,y):
    return x/y * 100

#the button to the bar chart page    
@app.route('/barchart',methods=['POST'])
def barchart():
    if request.method == 'POST':
        dataset = pd.read_csv('predicted_ubertweets.csv')
        #create new data frames for all sentiments
        tweet_negative = dataset[dataset["Uber_predicted_label"] == 0]
        tweet_positive = dataset[dataset["Uber_predicted_label"] == 1]
        #function for calculating the percentage of all the sentiments
        pos_per = calc_percentage(len(tweet_positive), len(dataset)), (len(tweet_positive))
        neg_per = calc_percentage(len(tweet_negative), len(dataset)), (len(tweet_negative))
        #the barchart
        
        return render_template('barchart.html', tweetspositive_text =(pos_per), tweetsnegative_text=(neg_per))
    else:
        return redirect(url_for("home"))
#the button to the pie chart page    
@app.route('/piechart',methods=['POST'])
def piechart():
    if request.method == 'POST':
        #create a bar graph by sentiment
        dataset = pd.read_csv('predicted_ubertweets.csv')
        tweet_negative = dataset[dataset["Uber_predicted_label"] == 0]
        tweet_positive = dataset[dataset["Uber_predicted_label"] == 1]
        
        #labels = dataset.groupby('Uber_predicted_label').count().index.values
        #values = dataset.groupby('Uber_predicted_label').size().values
        #plt.bar(labels, values)

        #create new data frames for all sentiments
        return render_template('piechart.html')
    else:
        return redirect(url_for("home"))

#

#creating the wordcloud
def create_wordcloud(text):
    stopwords = set(STOPWORDS)
    wc = WordCloud(background_color = "white", max_words = 3000, stopwords = stopwords, repeat = True)
    wc.generate(str(text))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis("off")
    plt.show()


@app.route('/wordcloudchart',methods=['POST'])
def wordcloudchart():
    if request.method == 'POST':
        dataset = pd.read_csv('predicted_ubertweets.csv')
        tweet_negative = dataset[dataset["Uber_predicted_label"] == 0]
        tweet_positive = dataset[dataset["Uber_predicted_label"] == 1]
        #word cloud for positive sentiments
        cloud1 = create_wordcloud(tweet_positive["tweet"].values)
        #wordcloud for negative sentiments
        cloud2 = create_wordcloud(tweet_negative["tweet"].values)
        
        return render_template('wordcloud.html', wordcloud_text=(cloud1), wordcloud2_text=(cloud2))
    else:
        return redirect(url_for("home"))


if __name__ == "__main__":
    app.run(debug=True) #if we encounter any error we can know what it is
