import dash
from dash import dcc, html
import pickle
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from dash.dependencies import Input, Output, State


""" Example logistic regression dataset
https://sushtend.com/machine-learning/datasets-for-practicing-logistic-regression/
"""

# Global variables
myheading1 = 'Predicting Mortgage Loan Approval'
image1 = 'ames_welcome.jpeg'
tabtitle = 'Mortgage Loans'
sourceurl = 'https://www.kaggle.com/burak3ergun/loan-data-set'
githublink = 'https://github.com/cahn1/504-mortgage-loans-predictor/tree' \
             '/update1'
approve_color = 'green'

# Model features
features = [
    'Credit_History',
    'LoanAmount',
    'Loan_Amount_Term',
    'ApplicantIncome',
    'CoapplicantIncome',
    'Property_Area',
    'Gender',
    'Education',
    'Self_Employed'
]


# Load pkl files
# dataframes for visualization
approved = pd.read_csv('model_components/approved_loans.csv')
denied = pd.read_csv('model_components/denied_loans.csv')

# random forest model
with open('model_components/loan_approval_rf_model.pkl', 'rb') as f:
    rf = pickle.load(f)

# encoder1
with open('model_components/loan_approval_onehot_encoder.pkl', 'rb') as f:
    encoder1 = pickle.load(f)

# ss_scaler1: monthly_return
with open('model_components/loan_approval_ss_scaler1.pkl', 'rb') as f:
    ss_scaler1 = pickle.load(f)

# ss_scaler2: ln_total_income
with open('model_components/loan_approval_total_income.pkl', 'rb') as f:
    ss_scaler2 = pickle.load(f)

# ss_scaler3: loan_amount
with open('model_components/loan_approval_loan_amount.pkl', 'rb') as f:
    ss_scaler3 = pickle.load(f)

# Load model comparison component
with open('model_components/model_comparison.pkl', 'rb') as f:
    model_comparison = pickle.load(f)


# Create a function that can take any 8 valid inputs & make a prediction
def make_predictions(args, threshold):
    try:
        # the order of the arguments must match the order of the features
        df = pd.DataFrame(columns=features)
        df.loc[0] = args

        # convert arguments from integers to floats:
        for i in ['Credit_History', 'LoanAmount', 'Loan_Amount_Term',
                  'ApplicantIncome', 'CoapplicantIncome']:
            df[i] = int(df[i])

        # recode a few columns using the same steps we employed on the
        # training data
        df['Gender'].replace({'Male': 1, 'Female': 0}, inplace=True)
        df['Education'].replace({'Graduate': 1, 'Not Graduate': 0}, inplace=True)
        df['Self_Employed'].replace({'Yes': 1, 'No': 0}, inplace=True)
        df['LoanAmount'] = df['LoanAmount'] * 1000

        # transform the categorical variable using the same encoder we
        # trained previously
        ohe = pd.DataFrame(encoder1.transform(df[['Property_Area']]).toarray())
        col_list = ['Property_Area_' + col for col
                    in ['Semiurban', 'Urban', 'Rural']]
        ohe.columns = col_list
        df = pd.concat([df, ohe], axis=1)

        # create new features using the scalers we trained earlier
        ln_monthly_return_raw = np.log(
            df['LoanAmount']/df['Loan_Amount_Term']).values
        ln_total_income_raw = np.log(
            int(df['ApplicantIncome']) + int(df['CoapplicantIncome']))
        ln_loanAmount_raw = np.log(1000 * df['LoanAmount'])
        df['ln_monthly_return'] = ss_scaler1.transform(
            np.array(ln_monthly_return_raw).reshape(-1, 1))
        df['ln_total_income'] = ss_scaler2.transform(
            np.array(ln_total_income_raw).reshape(-1, 1))
        df['ln_LoanAmount'] = ss_scaler3.transform(
            np.array(ln_loanAmount_raw).reshape(-1, 1))

        # drop & rearrange the columns in the order expected by your trained
        # model!
        df = df[['Gender', 'Education', 'Self_Employed', 'Credit_History',
                 'Property_Area_Semiurban', 'Property_Area_Urban',
                 'Property_Area_Rural', 'ln_monthly_return',
                 'ln_total_income', 'ln_LoanAmount'
        ]]

        prob = rf.predict_proba(df)
        raw_approval_prob = prob[0][1]
        threshold = threshold * .01
        approval_func = lambda y: 'Approved' \
            if raw_approval_prob > threshold else 'Denied'
        formatted_denial_prob = "{:,.1f}%".format(100 * prob[0][0])
        formatted_approval_prob = "{:,.1f}%".format(100 * prob[0][1])

        # return list(df.columns), list(df.columns), str(df.head().values)
        result = approval_func(raw_approval_prob)
        result_color = {'color': '#1abc9c'} if result == 'Approved' \
            else {'color': '#cb4335'}
        return result, result_color, formatted_approval_prob, \
               formatted_denial_prob
    except Exception as e:
        return 'Invalid inputs','Invalid inputs','Invalid inputs'


# Visualization
def make_loans_cube(*args):
    newdata = pd.DataFrame([args[:9]], columns=features)
    newdata['Combined_Income'] = \
        newdata['ApplicantIncome'] + newdata['CoapplicantIncome']

    trace0=go.Scatter3d(
        x=approved['LoanAmount'],
        y=approved['Combined_Income'],
        z=approved['Loan_Amount_Term'],
        name='Approved',
        mode='markers',
        text=list(zip(
            ["Credit: {}".format(x) for x in approved['Credit_History']],
            ["<br>Education: {}".format(x) for x in approved['Education']],
            ["<br>Property Area: {}".format(x) for x in approved['Property_Area']],
            ["<br>Gender: {}".format(x) for x in approved['Gender']],
            ["<br>Education: {}".format(x) for x in approved['Education']],
            ["<br>Self-Employed: {}".format(x) for x in approved['Self_Employed']]
        )),
        hovertemplate =
            '<b>Loan Amount: $%{x:.0f}K</b>' +
            '<br><b>Income: $%{y:.0f}</b>' +
            '<br><b>Term: %{z:.0f}</b>' +
            '<br>%{text}',
        hoverinfo='text',
        surfacecolor='#e5e7e9',
        marker=dict(size=6, color='#45b39d', opacity=0.4))

    trace1=go.Scatter3d(
        x=denied['LoanAmount'],
        y=denied['Combined_Income'],
        z=denied['Loan_Amount_Term'],
        name='Denied',
        mode='markers',
        text=list(zip(
            ["Credit: {}".format(x) for x in denied['Credit_History']],
            ["<br>Education: {}".format(x) for x in denied['Education']],
            ["<br>Property Area: {}".format(x) for x in denied['Property_Area']],
            ["<br>Gender: {}".format(x) for x in denied['Gender']],
            ["<br>Education: {}".format(x) for x in denied['Education']],
            ["<br>Self-Employed: {}".format(x) for x in denied['Self_Employed']]
        )),
        hovertemplate =
            '<b>Loan Amount: $%{x:.0f}K</b>' +
            '<br><b>Income: $%{y:.0f}</b>' +
            '<br><b>Term: %{z:.0f}</b>' +
            '<br>%{text}',
        hoverinfo='text',
        surfacecolor='#e5e7e9',
        marker=dict(size=6, color='#f39c12', opacity=0.4))

    trace2=go.Scatter3d(
        x=newdata['LoanAmount'],
        y=newdata['Combined_Income'],
        z=newdata['Loan_Amount_Term'],
        name='Applicant',
        mode='markers',
        text=list(zip(
            ["Credit: {} ".format(x) for x in newdata['Credit_History']],
            ["<br>Education: {} ".format(x) for x in newdata['Education']],
            ["<br>Property Area: {}".format(x) for x in newdata['Property_Area']],
            ["<br>Gender: {}".format(x) for x in newdata['Gender']],
            ["<br>Education: {}".format(x) for x in newdata['Education']],
            ["<br>Self-Employed: {}".format(x) for x in newdata['Self_Employed']]
        )),
        hovertemplate =
            '<b>Loan Amount: $%{x:.0f}K</b>' +
            '<br><b>Income: $%{y:.0f}</b>' +
            '<br><b>Term: %{z:.0f}</b>' +
            '<br>%{text}',
        hoverinfo='text',
        surfacecolor='#e5e7e9',
        marker=dict(
            size=5, color='#2e86c1', line=dict(width=0.5, color='#243bdc')))

    layout = go.Layout(
        title="Loan Status",
        showlegend=True,
        plot_bgcolor='#e5e7e9',
        scene=dict(
        xaxis=dict(title='Loan Amount'),
        yaxis=dict(title='Combined Income'),
        zaxis=dict(title='Term')
    ))
    fig = go.Figure([trace0, trace1, trace2], layout)
    return fig


# app server config
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server
app.title = tabtitle

# Set up Dash layout
app.layout = html.Div([
    html.H1(myheading1, style={
        'textAlign': 'center', 'margin': '48px 0', 'fontFamily': 'system-ui'}),
    dcc.Tabs(id="tabs", children=[
        dcc.Tab(label='Predicting Mortgage Loan Approval', children=[
            html.Div(
                className='two columns',
                children=[
                    html.H3("Features"),
                    html.Div('Credit History'),
                    dcc.Input(id='Credit_History', value=1, type='number', min=0, max=1, step=1),
                    html.Div('Loan Amount (thousands)'),
                    dcc.Input(id='LoanAmount', value=100, type='number', min=10, max=250, step=10),
                    html.Div('Term (months)'),
                    dcc.Input(id='Loan_Amount_Term', value=360, type='number', min=120, max=480, step=50),
                    html.Div('Applicant Monthly Income'),
                    dcc.Input(id='ApplicantIncome', value=5500, type='number', min=100, max=6000, step=100),
                    html.Div('Co-Applicant Monthly Income'),
                    dcc.Input(id='CoapplicantIncome', value=2500, type='number', min=0, max=6000, step=100),
                    html.Div('Property Area'),
                    dcc.Dropdown(
                        id='Property_Area',
                        options=[{'label': i, 'value': i} for i in ['Semiurban','Urban','Rural']],
                        value='Urban'),
                    html.Div('Gender'),
                    dcc.Dropdown(
                        id='Gender',
                        options=[{'label': i, 'value': i} for i in ['Male', 'Female']],
                        value='Female'),
                    html.Div('Education'),
                    dcc.Dropdown(
                        id='Education',
                        options=[{'label': i, 'value': i} for i in ['Graduate', 'Not Graduate']],
                        value='Graduate'),
                    html.Div('Self Employed'),
                    dcc.Dropdown(
                        id='Self_Employed',
                        options=[{'label': i, 'value': i} for i in ['No','Yes']],
                        value='No'),
                    html.Div('Approval Threshold'),
                    dcc.Input(id='Threshold', value=50, type='number', min=0, max=100, step=1),
            ], ),
            html.Div(
                className='two columns',
                children=[
                    html.H3('Predictions'),
                    html.Button(children='Submit', id='submit-val', n_clicks=0,
                                style={
                                    'background-color': 'red',
                                    'color': 'white',
                                    'margin-left': '5px',
                                    'verticalAlign': 'center',
                                    'horizontalAlign': 'center'}
                                ),
                    html.Div('Predicted Status:'),
                    html.B(children=[
                        html.Div(
                            id='PredResults',
                            style={'color': approve_color}),
                    ]),
                    html.Br(),
                    html.Div('Probability of Approval:'),
                    html.Div(id='ApprovalProb'),
                    html.Br(),
                    html.Div('Probability of Denial:'),
                    html.Div(id='DenialProb')
                ], ),
            html.Div(
                className='eight columns',
                children=[
                    html.Div(children=[
                        html.P('Updated:'),
                        html.Li('Updated imbalanced data using Random over-sampling with imblearn'),
                        html.A('Random over-sampling with imblearn',
                               href='https://www.analyticsvidhya.com/blog/2020/07/10-techniques-to-deal-with-class-imbalance-in-machine-learning/', target='_blank'),
                        dcc.Markdown('''
                        ```python
                            X=loans_df.drop(columns=['Loan_Status', 'Combined_Income'])
                            y=loans_df['Loan_Status']
                            
                            from imblearn.over_sampling import RandomOverSampler
                            tl = RandomOverSampler(sampling_strategy='auto')
                            X_tl, y_tl = tl.fit_resample(X, y)
                            
                            print(f'X: {X_tl.shape}, y: {y_tl.shape}')
                            X: (664, 10), y: (664,)
                            X_train, X_test, y_train, y_test = train_test_split(
                                X_tl, y_tl, test_size=.3, random_state=12, stratify=y_tl)                        
                        ```''')
                    ]),
                    html.Hr(),
                    html.Div(
                        children=[dcc.Graph(id='fig1',style={'width': '90vh', 'height': '90vh'}),],
                    ),
                    html.Div([
                        dcc.Graph(figure=model_comparison, id='model_comp')]),
                ]),
        ], className='twelve columns',),
        # dcc.Tab(label='Tab two', children=[
        # ]),
    ],
    style={'fontFamily': 'system-ui'},
    content_style={
        'borderLeft': '1px solid #d6d6d6',
        'borderRight': '1px solid #d6d6d6',
        'borderBottom': '1px solid #d6d6d6',
        'padding': '10px'
    },
    parent_style={
        'maxWidth': '1600px',
        'margin': '0 auto'
    })
])


# callback: Predictions
@app.callback(
    Output(component_id='PredResults', component_property='children'),
    Output(component_id='PredResults', component_property='style'),
    Output(component_id='ApprovalProb', component_property='children'),
    Output(component_id='DenialProb', component_property='children'),

    State(component_id='Credit_History', component_property='value'),
    State(component_id='LoanAmount', component_property='value'),
    State(component_id='Loan_Amount_Term', component_property='value'),
    State(component_id='ApplicantIncome', component_property='value'),
    State(component_id='CoapplicantIncome', component_property='value'),
    State(component_id='Property_Area', component_property='value'),
    State(component_id='Gender', component_property='value'),
    State(component_id='Education', component_property='value'),
    State(component_id='Self_Employed', component_property='value'),
    State(component_id='Threshold', component_property='value'),

    Input(component_id='submit-val', component_property='n_clicks'),
)
def func(*args):
    params = [arg for arg in args[:9]]
    return make_predictions(params, args[9])


# callback: Visualization
@app.callback(
    Output(component_id='fig1', component_property='figure'),

    State(component_id='Credit_History', component_property='value'),
    State(component_id='LoanAmount', component_property='value'),
    State(component_id='Loan_Amount_Term', component_property='value'),
    State(component_id='ApplicantIncome', component_property='value'),
    State(component_id='CoapplicantIncome', component_property='value'),
    State(component_id='Property_Area', component_property='value'),
    State(component_id='Gender', component_property='value'),
    State(component_id='Education', component_property='value'),
    State(component_id='Self_Employed', component_property='value'),

    Input(component_id='submit-val', component_property='n_clicks'),
)
def visualize(*args):
    return make_loans_cube(*args)


if __name__ == '__main__':
    app.run_server(debug=True)
