import time
import os
from flask import Flask, render_template, request, redirect, url_for, session, jsonify
from matplotlib import pyplot as plt
from werkzeug.utils import secure_filename
import pandas as pd
import plotly.express as px
import scipy.stats
from scipy import stats
from datetime import datetime
import seaborn as sns
import numpy as np
from scipy.stats import f
from statsmodels.formula.api import ols


app = Flask(__name__)
app.secret_key = '14136501411'
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv', 'excel', 'xlsx'}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/")
def index():
    return render_template('index.html')


@app.route('/list_files')
def list_files():
    try:
        files = []
        for filename in os.listdir(app.config['UPLOAD_FOLDER']):
            if allowed_file(filename):
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file_info = {
                    'name': filename,
                    'size': f"{os.path.getsize(filepath) / (1024):.2f} KB",
                    'date': datetime.fromtimestamp(os.path.getmtime(filepath)).strftime('%Y-%m-%d %H:%M:%S')
                }
                files.append(file_info)
        return jsonify({'files': files})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        selected_file = request.form.get('selected_file')
        if selected_file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], selected_file)
            if os.path.exists(filepath):
                session['uploaded_file'] = filepath
                return '', 200
        return 'No file part', 400

    file = request.files['file']

    if file.filename == '':
        return 'No selected file', 400

    if file and allowed_file(file.filename):
        # Check file size
        file.seek(0, os.SEEK_END)
        file_length = file.tell()
        if file_length > MAX_FILE_SIZE:
            return 'File too large', 400
        file.seek(0)

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        try:
            file.save(filepath)
            session['uploaded_file'] = filepath
            return '', 200
        except Exception as e:
            return f'Error saving file: {str(e)}', 500

    return 'Invalid file type', 400


@app.route("/dashboard")
def dashboard():
    if 'uploaded_file' not in session:
        return redirect(url_for('index'))

    filepath = session['uploaded_file']

    if not os.path.exists(filepath):
        return redirect(url_for('index'))

    try:
        operations_count = 0
        start_time = time.time()

        # Read the uploaded CSV file
        df = pd.read_csv(filepath)
        operations_count += 1
        total_rows = len(df)
        duplicates_count = df.duplicated().sum()
        null_values = df.isnull().sum()
        operations_count += 1
        columns_count = len(df.columns)
        operations_count += 1
        filtered_rows = total_rows - duplicates_count
        removed_rows = total_rows - filtered_rows
        df = df.drop_duplicates()
        time_elapsed = round(time.time() - start_time, 2)
        df['Extracurricular_Encoded'] = df['Extracurricular Activities'].map({'Yes': 1, 'No': 0})
        operations_count += 1
        data_summary = df.describe().to_dict()
        File_Size = round(os.path.getsize(filepath) / (1024 * 1024), 4)

        plt.figure(figsize=(15, 8))
        sns.boxplot(data=df, palette="Set2")
        plt.title("Data Distribution")
        plt.savefig("Static/Plots/boxplot.png")

        clean_code = '''
         operations_count = 0
        start_time = time.time()

        # Read the uploaded CSV file
        df = pd.read_csv(filepath)
        operations_count += 1
        total_rows = len(df)
        duplicates_count = df.duplicated().sum()
        null_values = df.isnull().sum()
        operations_count += 1
        columns_count = len(df.columns)
        operations_count += 1
        filtered_rows = total_rows - duplicates_count
        removed_rows = total_rows - filtered_rows
        df = df.drop_duplicates()
        time_elapsed = round(time.time() - start_time, 2)
        df['Extracurricular_Encoded'] = df['Extracurricular Activities'].map({'Yes': 1, 'No': 0})
        operations_count += 1
        data_summary = df.describe().to_dict()
        File_Size = round(os.path.getsize(filepath) / (1024 * 1024), 4)
        
        plt.figure(figsize=(15, 8))
        sns.boxplot(data=df, palette="Set2")
        plt.title("Data Distribution")
        plt.savefig("Static/Plots/boxplot.png")
        '''

        fig1 = px.box(df, x='Extracurricular Activities', y='Performance Index',
                      title='Score Distribution by Extracurricular Activities',
                      color='Extracurricular Activities',
                      labels={'Performance Index': 'Performance Index',
                              'Extracurricular Activities': 'Extracurricular Activities'})

        fig1.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            title_font=dict(color='white')
        )

        fig2 = px.scatter(df, x='Hours Studied', y='Performance Index',
                          color='Extracurricular Activities',
                          title='Hours Studied vs Performance by Extracurricular Activities',
                          hover_data=['Previous Scores', 'Sleep Hours', 'Sample Question Papers Practiced'],
                          labels={'Hours Studied': 'Hours Studied',
                                  'Performance Index': 'Performance Index',
                                  'Extracurricular Activities': 'Extracurricular Activities'})

        fig2.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            title_font=dict(color='white')
        )

        _code2_ = '''
        fig1 = px.box(df, x='Extracurricular Activities', y='Performance Index',
                      title='Score Distribution by Extracurricular Activities',
                      color='Extracurricular Activities',
                      labels={'Performance Index': 'Performance Index',
                              'Extracurricular Activities': 'Extracurricular Activities'})

        fig1.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            title_font=dict(color='white')
        )

        fig2 = px.scatter(df, x='Hours Studied', y='Performance Index',
                          color='Extracurricular Activities',
                          title='Hours Studied vs Performance by Extracurricular Activities',
                          hover_data=['Previous Scores', 'Sleep Hours', 'Sample Question Papers Practiced'],
                          labels={'Hours Studied': 'Hours Studied',
                                  'Performance Index': 'Performance Index',
                                  'Extracurricular Activities': 'Extracurricular Activities'})

        fig2.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            title_font=dict(color='white')
        )
        '''

        fig3 = px.bar(df, x='Sleep Hours', y='Performance Index',
                      color='Extracurricular Activities', barmode='group',
                      title='Average Performance by Sleep Hours and Extracurricular Activities',
                      labels={'Performance Index': 'Performance Index',
                              'Sleep Hours': 'Sleep Hours',
                              'Extracurricular Activities': 'Extracurricular Activities'})

        fig3.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            title_font=dict(color='white')
        )

        fig4 = px.violin(df, x='Sample Question Papers Practiced', y='Performance Index',
                         color='Extracurricular Activities', box=True,
                         title='Effect of Practicing Sample Questions on Performance',
                         labels={'Performance Index': 'Performance Index',
                                 'Sample Question Papers Practiced': 'Sample Papers Practiced',
                                 'Extracurricular Activities': 'Extracurricular Activities'})

        fig4.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            title_font=dict(color='white')
        )

        _code3_ = '''
         fig3 = px.bar(df, x='Sleep Hours', y='Performance Index',
                      color='Extracurricular Activities', barmode='group',
                      title='Average Performance by Sleep Hours and Extracurricular Activities',
                      labels={'Performance Index': 'Performance Index',
                              'Sleep Hours': 'Sleep Hours',
                              'Extracurricular Activities': 'Extracurricular Activities'})

        fig3.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            title_font=dict(color='white')
        )

        fig4 = px.violin(df, x='Sample Question Papers Practiced', y='Performance Index',
                         color='Extracurricular Activities', box=True,
                         title='Effect of Practicing Sample Questions on Performance',
                         labels={'Performance Index': 'Performance Index',
                                 'Sample Question Papers Practiced': 'Sample Papers Practiced',
                                 'Extracurricular Activities': 'Extracurricular Activities'})

        fig4.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            title_font=dict(color='white')
        )
        '''

        plt.figure(figsize=(10, 6))
        sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm', fmt=".2f")
        plt.title("Correlation Heatmap")

        df['Sleep Category'] = pd.cut(df['Sleep Hours'],
                                      bins=[0, 5, 7, 9, 24],
                                      labels=['<5 hours', '5-7 hours', '7-9 hours', '>9 hours'])

        fig5 = px.sunburst(df, path=['Sleep Category', 'Extracurricular Activities'],
                           values='Performance Index',
                           title='Performance Distribution by Sleep Hours and Extracurricular Activities',
                           labels={'Sleep Category': 'Sleep Hours',
                                   'Extracurricular Activities': 'Extracurricular Activities',
                                   'Performance Index': 'Performance Index'})

        fig5.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            title_font=dict(color='white')
        )
        _code4_ = '''
        
        plt.figure(figsize=(10, 6))
        sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm', fmt=".2f")
        plt.title("Correlation Heatmap")
        plt.show()

        df['Sleep Category'] = pd.cut(df['Sleep Hours'],
                                      bins=[0, 5, 7, 9, 24],
                                      labels=['<5 hours', '5-7 hours', '7-9 hours', '>9 hours'])

        fig5 = px.sunburst(df, path=['Sleep Category', 'Extracurricular Activities'],
                           values='Performance Index',
                           title='Performance Distribution by Sleep Hours and Extracurricular Activities',
                           labels={'Sleep Category': 'Sleep Hours',
                                   'Extracurricular Activities': 'Extracurricular Activities',
                                   'Performance Index': 'Performance Index'})

        fig5.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            title_font=dict(color='white')
        )
        '''

        plot_html1 = fig1.to_html(full_html=False, include_plotlyjs='cdn', config={'responsive': True})
        plot_html2 = fig2.to_html(full_html=False, include_plotlyjs='cdn', config={'responsive': True})
        plot_html3 = fig3.to_html(full_html=False, include_plotlyjs='cdn', config={'responsive': True})
        plot_html4 = fig4.to_html(full_html=False, include_plotlyjs='cdn', config={'responsive': True})
        plot_html5 = fig5.to_html(full_html=False, include_plotlyjs='cdn', config={'responsive': True})

        return render_template('dashboard.html',
                               plot_html1=plot_html1,
                               plot_html2=plot_html2,
                               plot_html3=plot_html3,
                               plot_html4=plot_html4,
                               plot_html5=plot_html5,
                               filename=os.path.basename(filepath),
                               total_rows=total_rows,
                               duplicates_count=duplicates_count,
                               null_values=null_values,
                               columns_count=columns_count,
                               removed_rows=removed_rows,
                               filtered_rows=filtered_rows,
                               elapsed_time=time_elapsed,
                               operations_count=operations_count,
                               data_summary=data_summary,
                               File_Size=File_Size,
                               clean_code = clean_code,
                               _code2_=_code2_,
                               _code3_=_code3_,
                               _code4_=_code4_,
                               )

    except Exception as e:
        return f"Error processing file: {str(e)}", 500

@app.route('/output')
def output():

    filepath = session['uploaded_file']
    df = pd.read_csv(filepath)
    df = df.drop_duplicates()
    df['Extracurricular_Encoded'] = df['Extracurricular Activities'].map({'Yes': 1, 'No': 0})

    def qname(name):
        return f'Q("{name}")'

    def partial_f_test(df, response, predictors):
        results = {}
        response_q = qname(response)
        predictors_q = [qname(p) for p in predictors]
        full_formula = f"{response_q} ~ {' + '.join(predictors_q)}"
        full_model = ols(full_formula, data=df).fit()
        RSS_full = sum((full_model.fittedvalues - df[response]) ** 2)
        df_full = full_model.df_resid

        for predictor in predictors:
            reduced_predictors = [p for p in predictors if p != predictor]
            reduced_q = [qname(p) for p in reduced_predictors]
            reduced_formula = f"{response_q} ~ {' + '.join(reduced_q)}"
            reduced_model = ols(reduced_formula, data=df).fit()
            RSS_reduced = sum((reduced_model.fittedvalues - df[response]) ** 2)
            df_reduced = reduced_model.df_resid

            num = (RSS_reduced - RSS_full) / (df_reduced - df_full)
            den = RSS_full / df_full
            F_stat = num / den

            alpha = 0.05
            dfn = df_reduced - df_full
            dfd = int(df_full)
            F_critical = f.ppf(1 - alpha, dfn, dfd)

            results[predictor] = {
                'Predictor': predictor,
                'f_stat': round(F_stat, 4),
                'f_critical': round(F_critical, 4),
                'decision': "Significant (keep)" if F_stat > F_critical else "Not Significant (can drop)",
                'is_significant': F_stat > F_critical
            }

        return results


    class MultipleLinearRegression:
        def __init__(self):
            self.F_stat = None
            self.SSR = None
            self.SST = None
            self.SSE = None
            self.y_bar = None
            self.n = None
            self.df_regression = None
            self.df_error = None
            self.B = None
            self.MSE = None
            self.r_squared = None
            self.error = None
            self.X = None
            self.y = None
            self.F_c = None
            self.R_2_adj = None

        def Fitting_Function(self, X, y):
            self.n = len(X)
            self.y_bar = np.mean(y)

            if isinstance(X, pd.DataFrame):
                X = X.copy()
                if 'B0' not in X.columns:
                    X.insert(0, 'B0', 1)
            else:
                X = np.column_stack([np.ones(self.n), X])

            X = np.array(X)
            y = np.array(y)

            XT_X = X.T @ X
            XT_y = X.T @ y
            self.B = np.linalg.inv(XT_X) @ XT_y

            self.SSE = y.T @ y - self.B.T @ XT_y
            self.SST = y.T @ y - self.n * (self.y_bar ** 2)
            self.SSR = self.SST - self.SSE
            self.r_squared = self.SSR / self.SST

            self.df_error = self.n - len(self.B)
            self.df_regression = len(self.B) - 1
            self.error = y - X @ self.B
            self.MSE = self.SSE / self.df_error
            self.X = X
            self.R_2_adj = 1 - ((self.n - 1) / (self.n - len(self.B)) * (self.SSE / self.SST))

        def Prediction_Model(self, X):
            if self.B is None:
                raise ValueError("Model not fitted yet.")

            if isinstance(X, pd.DataFrame):
                X = X.copy()
                if 'B0' not in X.columns:
                    X.insert(0, 'B0', 1)
            else:
                # Take a sequence of 1-D arrays and stack them as columns to make a single 2-D array. 2-D arrays are stacked as-is, just like with hstack. 1-D arrays are turned into 2-D columns first.
                X = np.column_stack([np.ones(len(X)), X])

            return X @ self.B

        def Plot_3D(self, X, y, feature_names=None):
            """Plot 3D regression for any two features while holding others constant"""
            if len(self.B) < 3:
                raise ValueError("Need at least two predictors for 3D plot")

            X = np.array(X)
            y = np.array(y)

            if feature_names is None:
                feature_names = [f'Feature {i}' for i in range(X.shape[1])]

            # Create subplots for all feature pairs
            from itertools import combinations
            from plotly.subplots import make_subplots
            import plotly.graph_objects as go

            feature_pairs = list(combinations(range(X.shape[1]), 2))
            rows = int(np.ceil(len(feature_pairs) / 2))

            fig = make_subplots(
                rows=rows, cols=2,
                specs=[[{'type': 'surface'}, {'type': 'surface'}] for _ in range(rows)],
                subplot_titles=[f"{feature_names[i]} vs {feature_names[j]}" for i, j in feature_pairs],
                horizontal_spacing=0.1,
                vertical_spacing=0.1
            )

            for idx, (i, j) in enumerate(feature_pairs):
                row = (idx // 2) + 1
                col = (idx % 2) + 1

                # Create grid for surface plot
                x1_range = np.linspace(X[:, i].min(), X[:, i].max(), 20)
                x2_range = np.linspace(X[:, j].min(), X[:, j].max(), 20)
                x1_grid, x2_grid = np.meshgrid(x1_range, x2_range)

                # Create prediction matrix (hold other features at their mean)
                X_grid = np.column_stack([np.ones(x1_grid.size)])
                for k in range(X.shape[1]):
                    if k == i:
                        X_grid = np.column_stack([X_grid, x1_grid.ravel()])
                    elif k == j:
                        X_grid = np.column_stack([X_grid, x2_grid.ravel()])
                    else:
                        X_grid = np.column_stack([X_grid, np.full(x1_grid.size, X[:, k].mean())])

                y_grid = (X_grid @ self.B).reshape(x1_grid.shape)

                fig.add_trace(
                    go.Scatter3d(
                        x=X[:, i], y=X[:, j], z=y,
                        mode='markers',
                        marker=dict(size=4, color='purple', opacity=0.7),
                        showlegend=False
                    ),
                    row=row, col=col
                )

                # Add surface plot
                fig.add_trace(
                    go.Surface(
                        x=x1_grid, y=x2_grid, z=y_grid,
                        colorscale='ice',
                        opacity=0.6,
                        showscale=False
                    ),
                    row=row, col=col
                )

                # Update subplot scene
                fig.update_scenes(
                    xaxis_title=feature_names[i],
                    yaxis_title=feature_names[j],
                    zaxis_title='Performance Index',
                    camera=dict(eye=dict(x=1.5, y=1.5, z=0.5)),
                    row=row, col=col,
                    xaxis=dict(color='white'),
                    yaxis=dict(color='white'),
                    zaxis=dict(color='white'),
                )

            fig.update_layout(
                title_text="Multiple Linear Regression - Feature Relationships",
                height=400 * rows,
                margin=dict(l=0, r=0, b=0, t=40),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                title_font=dict(color='white')
            )
            return fig.to_html(full_html=False, include_plotlyjs='cdn', config={'responsive': True})

        def Anova_Table(self):
            if self.B is None:
                raise ValueError("The model has not been fitted yet.")

            self.df_regression = len(self.B) - 1
            df_total = self.n - 1

            MSR = self.SSR / self.df_regression
            MSE = self.SSE / self.df_error

            self.F_stat = MSR / MSE

            anova_data = {
                'Source': ['Regression', 'Error', 'Total'],
                'Sum of Squares': [self.SSR, self.SSE, self.SST],
                'Degrees of Freedom': [self.df_regression, self.df_error, df_total],
                'Mean Square': [MSR, MSE, ""],
                'F-Statistic': [self.F_stat, "", ""]
            }

            anova_table = pd.DataFrame(anova_data)
            return anova_table

        def hypothesis_test(self, alpha=0.05):
            self.F_c = scipy.stats.f.ppf(1 - alpha, self.df_regression, self.df_error)

            if self.F_stat > self.F_c:
                conclusion = ("Since F₀ > F꜀, we reject the null hypothesis.\n"
                              "Therefore, there's a relationship between x and y.")
            else:
                conclusion = ("Since F꜀ > F₀, we don't reject the null hypothesis.\n"
                              "Therefore, there's no relationship between x and y.")

            html_output = {
                "I1": "β₁ = β₂ = ... = βₙ = 0 (No relationship) ",
                "I2": "At Least one Coefficient βᵢ ≠ 0 for i = 1,2,...,n (Relationship exists)",
                "I3": [self.F_stat, self.F_c],
                "I4": conclusion,
            }

            return html_output

        def Interval_Estimation(self, alpha=0.05, Features_name=None, sigma=None):
            if self.B is None or self.MSE is None:
                raise ValueError("Fit the model before running interval estimation.")

            XTX_inv = np.linalg.inv(self.X.T @ self.X)

            if sigma is None:
                std_errors = np.sqrt(np.diag(self.MSE * XTX_inv))
                critical_value = stats.t.ppf(1 - alpha / 2, df=self.df_error)
            else:
                std_errors = np.sqrt(np.diag((sigma ** 2) * XTX_inv))
                critical_value = stats.norm.ppf(1 - alpha / 2)

            lower_bounds = self.B.flatten() - critical_value * std_errors
            upper_bounds = self.B.flatten() + critical_value * std_errors

            if Features_name is None:
                terms = ['Intercept β₀'] + [f'x{i}' for i in range(1, len(self.B.flatten()))]
            else:
                terms = ['Intercept β₀'] + Features_name

            return pd.DataFrame({
                'Term': terms,
                'Coefficient': self.B.flatten(),
                'Lower Bound': lower_bounds,
                'Upper Bound': upper_bounds,
                'Confidence Level': [f'{(1 - alpha) * 100:.1f}%' for _ in self.B.flatten()]
            })

    MLR_CODE = '''
    class MultipleLinearRegression:
        def __init__(self):
            self.F_stat = None
            self.SSR = None
            self.SST = None
            self.SSE = None
            self.y_bar = None
            self.n = None
            self.df_regression = None
            self.df_error = None
            self.B = None
            self.MSE = None
            self.r_squared = None
            self.error = None
            self.X = None
            self.y = None
            self.F_c = None
            self.R_2_adj = None

        def Fitting_Function(self, X, y):
            self.n = len(X)
            self.y_bar = np.mean(y)

            if isinstance(X, pd.DataFrame):
                X = X.copy()
                if 'B0' not in X.columns:
                    X.insert(0, 'B0', 1)
            else:
                X = np.column_stack([np.ones(self.n), X])

            X = np.array(X)
            y = np.array(y)

            XT_X = X.T @ X
            XT_y = X.T @ y
            self.B = np.linalg.inv(XT_X) @ XT_y

            self.SSE = y.T @ y - self.B.T @ XT_y
            self.SST = y.T @ y - self.n * (self.y_bar ** 2)
            self.SSR = self.SST - self.SSE
            self.r_squared = self.SSR / self.SST

            self.df_error = self.n - len(self.B)
            self.df_regression = len(self.B) - 1
            self.error = y - X @ self.B
            self.MSE = self.SSE / self.df_error
            self.X = X
            self.R_2_adj = 1 - ((self.n - 1) / (self.n - len(self.B)) * (self.SSE / self.SST))

        def Prediction_Model(self, X):
            if self.B is None:
                raise ValueError("Model not fitted yet.")

            if isinstance(X, pd.DataFrame):
                X = X.copy()
                if 'B0' not in X.columns:
                    X.insert(0, 'B0', 1)
            else:
                X = np.column_stack([np.ones(len(X)), X])

            return X @ self.B

        def Plot_3D(self, X, y, feature_names=None):
            """Plot 3D regression for any two features while holding others constant"""
            if len(self.B) < 3:
                raise ValueError("Need at least two predictors for 3D plot")

            X = np.array(X)
            y = np.array(y)

            if feature_names is None:
                feature_names = [f'Feature {i}' for i in range(X.shape[1])]

            # Create subplots for all feature pairs
            from itertools import combinations
            from plotly.subplots import make_subplots
            import plotly.graph_objects as go

            feature_pairs = list(combinations(range(X.shape[1]), 2))
            rows = int(np.ceil(len(feature_pairs) / 2))

            fig = make_subplots(
                rows=rows, cols=2,
                specs=[[{'type': 'surface'}, {'type': 'surface'}] for _ in range(rows)],
                subplot_titles=[f"{feature_names[i]} vs {feature_names[j]}" for i, j in feature_pairs],
                horizontal_spacing=0.1,
                vertical_spacing=0.1
            )

            for idx, (i, j) in enumerate(feature_pairs):
                row = (idx // 2) + 1
                col = (idx % 2) + 1

                # Create grid for surface plot
                x1_range = np.linspace(X[:, i].min(), X[:, i].max(), 20)
                x2_range = np.linspace(X[:, j].min(), X[:, j].max(), 20)
                x1_grid, x2_grid = np.meshgrid(x1_range, x2_range)

                # Create prediction matrix (hold other features at their mean)
                X_grid = np.column_stack([np.ones(x1_grid.size)])
                for k in range(X.shape[1]):
                    if k == i:
                        X_grid = np.column_stack([X_grid, x1_grid.ravel()])
                    elif k == j:
                        X_grid = np.column_stack([X_grid, x2_grid.ravel()])
                    else:
                        X_grid = np.column_stack([X_grid, np.full(x1_grid.size, X[:, k].mean())])

                y_grid = (X_grid @ self.B).reshape(x1_grid.shape)

                fig.add_trace(
                    go.Scatter3d(
                        x=X[:, i], y=X[:, j], z=y,
                        mode='markers',
                        marker=dict(size=4, color='purple', opacity=0.7),
                        showlegend=False
                    ),
                    row=row, col=col
                )

                # Add surface plot
                fig.add_trace(
                    go.Surface(
                        x=x1_grid, y=x2_grid, z=y_grid,
                        colorscale='ice',
                        opacity=0.6,
                        showscale=False
                    ),
                    row=row, col=col
                )

                # Update subplot scene
                fig.update_scenes(
                    xaxis_title=feature_names[i],
                    yaxis_title=feature_names[j],
                    zaxis_title='Performance Index',
                    camera=dict(eye=dict(x=1.5, y=1.5, z=0.5)),
                    row=row, col=col,
                    xaxis=dict(color='white'),
                    yaxis=dict(color='white'),
                    zaxis=dict(color='white'),
                )

            fig.update_layout(
                title_text="Multiple Linear Regression - Feature Relationships",
                height=400 * rows,
                margin=dict(l=0, r=0, b=0, t=40),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                title_font=dict(color='white')
            )
            return fig.to_html(full_html=False, include_plotlyjs='cdn', config={'responsive': True})

        def Anova_Table(self):
            if self.B is None:
                raise ValueError("The model has not been fitted yet.")

            self.df_regression = len(self.B) - 1
            df_total = self.n - 1

            MSR = self.SSR / self.df_regression
            MSE = self.SSE / self.df_error

            self.F_stat = MSR / MSE

            anova_data = {
                'Source': ['Regression', 'Error', 'Total'],
                'Sum of Squares': [self.SSR, self.SSE, self.SST],
                'Degrees of Freedom': [self.df_regression, self.df_error, df_total],
                'Mean Square': [MSR, MSE, ""],
                'F-Statistic': [self.F_stat, "", ""]
            }

            anova_table = pd.DataFrame(anova_data)
            return anova_table

        def hypothesis_test(self, alpha=0.05):
            self.F_c = scipy.stats.f.ppf(1 - alpha, self.df_regression, self.df_error)

            if self.F_stat > self.F_c:
                conclusion = ("Since F₀ > F꜀, we reject the null hypothesis.\n"
                              "Therefore, there's a relationship between x and y.")
            else:
                conclusion = ("Since F꜀ > F₀, we don't reject the null hypothesis.\n"
                              "Therefore, there's no relationship between x and y.")

            html_output = {
                "I1": "β₁ = β₂ = ... = βₙ = 0 (No relationship) ",
                "I2": "At Least one Coefficient βᵢ ≠ 0 for i = 1,2,...,n (Relationship exists)",
                "I3": [self.F_stat, self.F_c],
                "I4": conclusion,
            }

            return html_output

        def Interval_Estimation(self, alpha=0.05, Features_name=None, sigma=None):
            if self.B is None or self.MSE is None:
                raise ValueError("Fit the model before running interval estimation.")

            XTX_inv = np.linalg.inv(self.X.T @ self.X)

            if sigma is None:
                std_errors = np.sqrt(np.diag(self.MSE * XTX_inv))
                critical_value = stats.t.ppf(1 - alpha / 2, df=self.df_error)
            else:
                std_errors = np.sqrt(np.diag((sigma ** 2) * XTX_inv))
                critical_value = stats.norm.ppf(1 - alpha / 2)

            lower_bounds = self.B.flatten() - critical_value * std_errors
            upper_bounds = self.B.flatten() + critical_value * std_errors

            if Features_name is None:
                terms = ['Intercept β₀'] + [f'x{i}' for i in range(1, len(self.B.flatten()))]
            else:
                terms = ['Intercept β₀'] + Features_name

            return pd.DataFrame({
                'Term': terms,
                'Coefficient': self.B.flatten(),
                'Lower Bound': lower_bounds,
                'Upper Bound': upper_bounds,
                'Confidence Level': [f'{(1 - alpha) * 100:.1f}%' for _ in self.B.flatten()]
            })
            #كدا الكلاس خلص, الي بعد كدا فانشكن جينيرال
            
    def qname(name):
        return f'Q("{name}")'

    def partial_f_test(df, response, predictors):
        results = {}
        response_q = qname(response)
        predictors_q = [qname(p) for p in predictors]
        full_formula = f"{response_q} ~ {' + '.join(predictors_q)}"
        full_model = ols(full_formula, data=df).fit()
        RSS_full = sum((full_model.fittedvalues - df[response]) ** 2)
        df_full = full_model.df_resid

        for predictor in predictors:
            reduced_predictors = [p for p in predictors if p != predictor]
            reduced_q = [qname(p) for p in reduced_predictors]
            reduced_formula = f"{response_q} ~ {' + '.join(reduced_q)}"
            reduced_model = ols(reduced_formula, data=df).fit()
            RSS_reduced = sum((reduced_model.fittedvalues - df[response]) ** 2)
            df_reduced = reduced_model.df_resid

            num = (RSS_reduced - RSS_full) / (df_reduced - df_full)
            den = RSS_full / df_full
            F_stat = num / den

            alpha = 0.05
            dfn = df_reduced - df_full
            dfd = int(df_full)
            F_critical = f.ppf(1 - alpha, dfn, dfd)

            results[predictor] = {
                'Predictor': predictor,
                'f_stat': round(F_stat, 4),
                'f_critical': round(F_critical, 4),
                'decision': "Significant (keep)" if F_stat > F_critical else "Not Significant (can drop)",
                'is_significant': F_stat > F_critical
            }

        return results  
        '''

    X3 = df[['Hours Studied', 'Previous Scores', 'Sleep Hours', 'Sample Question Papers Practiced','Extracurricular_Encoded']]
    y3 = df['Performance Index']
    features3 = ['Hours Studied β₁', 'Previous Scores β₂', 'Sleep Hours β₃', 'Sample Papers β₄','Extracurricular Activities β₅']
    MLR3 = MultipleLinearRegression()
    MLR3.Fitting_Function(X3, y3)
    B0 = round(MLR3.B[0], 2)
    B1 = round(MLR3.B[1], 2)
    B2 = round(MLR3.B[2], 2)
    B3 = round(MLR3.B[3], 2)
    B4 = round(MLR3.B[4], 2)
    B5 = round(MLR3.B[5], 2)
    ANOVA = MLR3.Anova_Table()
    R2 = round(MLR3.r_squared, 6)
    R_2_adj = round(MLR3.R_2_adj, 6)
    Predicted = MLR3.Prediction_Model(X3)
    F_critical = MLR3.F_c
    _3D_PLOT = MLR3.Plot_3D(X3, y3, features3)
    Hypotheses_code = MLR3.hypothesis_test()
    Comparison = pd.DataFrame({"Actual": y3, "Predicted": Predicted})
    Inter = MLR3.Interval_Estimation(alpha=0.05, Features_name=features3)

    predictors = ['Hours Studied', 'Previous Scores', 'Extracurricular Activities', 'Sleep Hours',
                  'Sample Question Papers Practiced']
    partial_f_results = partial_f_test(df, 'Performance Index', predictors)
    estimator_names = list(partial_f_results.keys())

    return render_template("Helper.html",
                               B0=B0,
                               B1=B1,
                               B2=B2,
                               B3=B3,
                               B4=B4,
                               B5 = B5,
                               ANOVA=ANOVA,
                              Hypotheses_code = Hypotheses_code,
                               R2=R2,
                               R_2_adj=R_2_adj,
                               Inter=Inter,
                               F_critical=F_critical,
                               _3D_PLOT = _3D_PLOT,
                               partial_f_results = partial_f_results,
                               estimator_names = estimator_names,
                               MLR_CODE = MLR_CODE
                           )

if __name__ == "__main__":
    app.run(debug=True)