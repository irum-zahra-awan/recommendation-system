# app.py
from flask import Flask, request, render_template # import main Flask class and request object
import pandas as pd
from rake_nltk import Rake
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
import re
app = Flask(__name__)  # create the Flask app
#app = Flask(__name__, template_folder='templates')
#########


#df_work.head()


# extract keywords using RAKE (NLP Library)
def extract_keywords(df_work):
    # Using RAKE(nlp lib) to extracct keywords from description and put them in new column
    df_work['Key_words'] = ""
    for index, row in df_work.iterrows():
        descrip = row['description']
        r = Rake()
        r.extract_keywords_from_text(descrip)
        key_words_dict_scores = r.get_word_degrees()
        row['Key_words'] = list(key_words_dict_scores.keys())


# Clean extracted keywords
def bag_of_words(df_work):
    df_work['bag_of_words'] = ''
    df_work['bag_of_words'] = [''.join(x) for x in df_work['Key_words'].map(lambda x: ' '.join(x).replace(',', ' '))]


# Clean data
def clean_data(df_work):
    # Coverting specialities to lower case, splitting on commas and removing spaces
    df_work['specialties'] = [','.join(x) for x in
                              df_work['specialties'].map(lambda x: x.lower().replace(' ', '').split(',')).values]
    df_work['industry'] = [','.join(x) for x in
                           df_work['industry'].map(lambda x: x.lower().replace(' ', '').split(',')).values]
    df_work['type'] = [','.join(x) for x in df_work['type'].map(lambda x: x.lower().replace(' ', '').split(',')).values]


# Process Input
def process_input(title, df_work):
    title = str(title).lower()
    title = title.strip() # remove leading and trailing spaces
    title = re.sub(' +', ' ',title) # remove duplicate spaces
    input_specs = set(df_work.loc[title, 'specialties'].split(','))
    input_industry = set(df_work.loc[title, 'industry'].split(','))
    input_type = set(df_work.loc[title, 'type'].split(','))

    return input_specs, input_industry, input_type


# Check matches by applying intersection
def interSection(input_, data):
    result = list(filter(lambda x: x in input_, data))
    return result


# assign Scores formula used: length-of-matched-specs / length-of-input-specs
def specs_scoring(df_work, input_specs):
    for index, row in df_work.iterrows():
        data = set(row['specialties'].split(','))
        result = interSection(input_specs, data)

        length_result = len(result)
        if length_result > 0:
            score = length_result / len(input_specs)
            row['spec_scores'] = score
        else:
            row['spec_scores'] = 0


def indutry_scoring(df_work, input_industry):
    for index, row in df_work.iterrows():
        data = set(row['industry'].split(','))
        result = interSection(input_industry, data)

        length_result = len(result)
        if length_result > 0:
            score = length_result / len(input_industry)
            row['industry_scores'] = score
        else:
            row['industry_scores'] = 0


def type_scoring(df_work, input_type):
    for index, row in df_work.iterrows():
        data = set(row['type'].split(','))
        result = interSection(input_type, data)

        length_result = len(result)
        if length_result > 0:
            score = length_result / len(input_type)
            row['type_scores'] = score
        else:
            row['type_scores'] = 0


def cosin_sim(title,df_work):
    title = str(title).lower()
    title = title.strip()  # remove leading and trailing spaces
    title = re.sub(' +', ' ', title)  # remove duplicate spaces
    # instantiating and generating the count matrix
    count = CountVectorizer()
    count_matrix = count.fit_transform(df_work['bag_of_words'])
    indices = pd.Series(df_work.index)
    # generating the cosine similarity matrix
    cosine_sim = cosine_similarity(count_matrix, count_matrix)
    # getting the index of the company that matches the title
    idx = indices[indices == title].index[0]
    # creating a Series with the similarity scores in descending order
    score_series = pd.Series(cosine_sim[idx])

    # Add scores in cosine_scores col
    for score, (index, row) in zip(score_series, df_work.iterrows()):
        row['cosine_scores'] = score

    return score_series


def calculate_total_score(df_work):
    for index, row in df_work.iterrows():
        sp_score = row['spec_scores']
        ind_score = row['industry_scores']
        typ_score = row['type_scores']
        des_score = row['cosine_scores']
        total = sp_score + ind_score + typ_score + des_score
        # print('total score: ', total)
        row['total_scores'] = total


def process_df(df_work):
    extract_keywords(df_work)
    bag_of_words(df_work)
    clean_data(df_work)
    df_work.drop(columns=[col for col in df_work.columns if col == ('Key_words')], inplace=True)


def calculate_all_scores(title, df_work):
    # process input
    input_specs, input_industry, input_type = process_input(title, df_work)
    # get scores
    df_work['spec_scores'] = ''
    specs_scoring(df_work, input_specs)
    df_work['industry_scores'] = ''
    indutry_scoring(df_work, input_industry)
    df_work['type_scores'] = ''
    type_scoring(df_work, input_type)
    df_work['cosine_scores'] = ''
    cosin_sim(title,df_work)
    df_work['total_scores'] = ''
    calculate_total_score(df_work)


def get_recommendations(df_work):
    sorted_series = pd.Series(df_work['total_scores'].sort_values(ascending=False))
    top_5_matches = list(sorted_series.iloc[1:6].index)
    return top_5_matches


#process_df(df_work)


# change title to any company you want to get recommendations for and run the script
# title = 'TOTAL'
# title = 'ENTREPOSE DRILLING'
# calculate_all_scores(title, df_work)
# result = get_recommendations(df_work)
# print('Top 5 Recommended companies: ', result)

# df_work = df_work[['spec_scores', 'industry_scores', 'type_scores', 'cosine_scores', 'total_scores']]
# print('Score Details of all the matches')

#result_df = df.loc[['Niko', 'Penelope']]
# for item in result:
# details = df_work.loc[[item]]
# print(details)

#########

@app.route('/company-recommendation', methods=['GET', 'POST'])
def formexample():
    if request.method == 'POST':  # this block is only entered when the form is submitted
        title = request.form.get('title')
        calculate_all_scores(title, df_work)
        result = get_recommendations(df_work)
        new_df = df_work.loc[[result[0], result[1], result[2], result[3], result[4]]]
        new_df.index = new_df.index.map(lambda x: str(x).upper())
        result = new_df[['spec_scores', 'industry_scores', 'type_scores', 'cosine_scores', 'total_scores','type', 'industry', 'specialties']]

        pd.set_option('display.max_colwidth', -1)


        #return render_template('result.html', result=result)
        return render_template("result.html", result=result.to_html())

        #return result
    return '<form method="POST"> Enter Company Name: <input type="text" name="title">' \
           '<br> <input type="submit" value="Submit"><br></form>'


@app.route('/json-example')
def jsonexample():
    return 'Todo...'


if __name__ == '__main__':
    # Read excel
    df_work = pd.read_excel("/home/irum/Desktop/Company_Similarity/Sample_Input_Companies.xlsx")
    # Drop Null values
    df_work.dropna(subset=['name', 'size_range', 'type', 'industry', 'specialties', 'description'], inplace=True)
    # Drop source and website col.we do not need it
    df_work.drop(columns=[col for col in df_work.columns if col == ('source')], inplace=True)
    df_work.drop(columns=[col for col in df_work.columns if col == ('website')], inplace=True)
    # Setting index
    df_work['name'] = df_work['name'].apply(lambda x: str(x).lower())
    df_work.set_index('name', inplace=True)
    df_work.to_excel("dataframe1.xlsx")

    process_df(df_work)

    app.run(debug=True, host='0.0.0.0', port=5000)  # run app in debug mode on port 5000
