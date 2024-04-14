from flask import Flask, render_template, request
import pandas as pd
from flask_table import Table, Col
from rake_nltk import Rake
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

# Building Flask table for showing recommendation results
class Item(Table):
    name = Col('Recommendations')
    description = Col('Score')

app = Flask(__name__)

# Rating Page
@app.route("/", methods=["GET", "POST"])
def rating():
    return render_template('welcome.html')

# Results Page
@app.route("/recommendation", methods=["GET", "POST"])
def recommendation():
    if request.method == 'POST':
        df = pd.read_csv('https://query.data.world/s/uikepcpffyo2nhig52xxeevdialfl7')
        df = df[['Title', 'Genre', 'Director', 'Actors', 'Plot']]
        # Discarding the commas between the actors' full names and getting only the first three names
        df['Actors'] = df['Actors'].map(lambda x: x.split(',')[:3])
        # Putting the genres in a list of words
        df['Genre'] = df['Genre'].map(lambda x: x.lower().split(','))
        df['Director'] = df['Director'].map(lambda x: x.split(' '))
        # Merging together first and last name for each actor and director, so it's considered as one word
        for index, row in df.iterrows():
            row['Actors'] = [x.lower().replace(' ', '') for x in row['Actors']]
            row['Director'] = ''.join(row['Director']).lower()
        # Initializing the new column
        df['Key_words'] = ""
        for index, row in df.iterrows():
            plot = row['Plot']
            # Instantiating Rake, by default it uses English stopwords from NLTK
            # and discards all punctuation characters
            r = Rake()
            # Extracting the words by passing the text
            r.extract_keywords_from_text(plot)
            # Getting the dictionary with key words and their scores
            key_words_dict_scores = r.get_word_degrees()
            # Assigning the key words to the new column
            row['Key_words'] = list(key_words_dict_scores.keys())

        # Dropping the Plot column
        df.drop(columns=['Plot'], inplace=True)
        df.set_index('Title', inplace=True)
        df['bag_of_words'] = ''
        columns = df.columns
        for index, row in df.iterrows():
            words = ''
            for col in columns:
                if col != 'Director':
                    words = words + ' '.join(row[col]) + ' '
                else:
                    words = words + row[col] + ' '
            row['bag_of_words'] = words

        df.drop(columns=[col for col in df.columns if col != 'bag_of_words'], inplace=True)
        # Instantiating and generating the count matrix
        count = CountVectorizer()
        count_matrix = count.fit_transform(df['bag_of_words'])
        # Creating a Series for the movie titles so they are associated to an ordered numerical
        # list I will use later to match the indexes
        indices = pd.Series(df.index)
        # Generating the cosine similarity matrix
        cosine_sim = cosine_similarity(count_matrix, count_matrix)

        def recommendations(title):
            recommended_movies = []
            # Getting the index of the movie that matches the title
            try:
                idx = indices[indices == title].index[0]
            except:
                res = "This movie is not registered in our database"
                return res, ['0']
            # Creating a Series with the similarity scores in descending order
            score_series = pd.Series(cosine_sim[idx]).sort_values(ascending=False)
            # Getting the indexes of the 10 most similar movies
            top_10_indexes = list(score_series.iloc[1:11].index)
            # Populating the list with the titles of the best 10 matching movies
            for i in top_10_indexes:
                recommended_movies.append(list(df.index)[i])
            return recommended_movies, score_series[1:11]

        int_features = [str(x) for x in request.form.values()]
        int_features = ''.join(int_features)
        output, score = recommendations(int_features)
        if output == "This movie is not registered in our database":
            return render_template('welcome.html', prediction_text=output)
        else:
            x = [str(i) for i in list(round(score, 3))]
            items = []
            for i in range(10):
                items.append(dict(name=output[i], description=x[i]))

            table = Item(items)
            table.border = True
            return render_template('welcome.html', prediction_text=table)


if __name__ == '__main__':
    app.run(debug=True)
