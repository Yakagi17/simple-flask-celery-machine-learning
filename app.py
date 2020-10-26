import os
import random
import time
from flask import Flask, request, render_template, session, flash, redirect, \
    url_for, jsonify
from celery import Celery

from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


app = Flask(__name__)
app.config['SECRET_KEY'] = 'top-secret!'


# Celery configuration
app.config['CELERY_BROKER_URL'] = 'amqp://guest@localhost//'
app.config['CELERY_RESULT_BACKEND'] = 'amqp'


# Initialize Celery
celery = Celery(app.name, broker=app.config['CELERY_BROKER_URL'], backend=app.config['CELERY_RESULT_BACKEND'])
# celery.conf.update(app.config)



@celery.task(bind=True)
def task_training_progress(self):
    """Background task that runs a long function with progress reports."""
    self.update_state(state='PROGRESS', meta={'current': 1, 'total': 5,'status': 'creating dependecy variable'})
    X,y = make_classification(n_samples=100, random_state=1)
    X_train, X_test, y_train, y_test = train_test_split(X,y, stratify=y, random_state=1)

    self.update_state(state='PROGRESS', meta={'current': 2, 'total': 5,'status': 'creating classification model'})
    clf = MLPClassifier(random_state=1, max_iter=500)

    self.update_state(state='PROGRESS', meta={'current': 3, 'total': 5,'status': 'training model'})
    clf.fit(X_train,y_train)

    self.update_state(state='PROGRESS', meta={'current': 4, 'total': 5,'status': 'predicting model with train and test data'})
    train_result = clf.predict(X_train)
    test_result = clf.predict(X_test)

    self.update_state(state='PROGRESS', meta={'current': 5, 'total': 5,'status': 'calculate model accuracy'})
    trian_accuracy = accuracy_score(y_train, train_result)
    test_accuracy = accuracy_score(y_test, test_result)
   
    
    return {'current': 100, 'total': 100, 'status': 'Task completed!',
            'accuracy': "Train Accuray : {} | Test Accuray : {}".format(str(trian_accuracy), str(test_accuracy))}

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'GET':
        return render_template('index.html')

    return redirect(url_for('index'))

@app.route('/training-progress', methods=['POST'])
def training_progress():
    task = task_training_progress.apply_async()
    return jsonify({}), 202, {'Location': url_for('taskstatus',
                                                  task_id=task.id)}


@app.route('/status/<task_id>')
def taskstatus(task_id):
    task = task_training_progress.AsyncResult(task_id)
    if task.state == 'PENDING':
        response = {
            'state': task.state,
            'current': 0,
            'total': 1,
            'status': 'Pending...'
        }
    elif task.state != 'FAILURE':
        response = {
            'state': task.state,
            'current': task.info.get('current', 0),
            'total': task.info.get('total', 1),
            'status': task.info.get('status', '')
        }
        if 'accuracy' in task.info:
            response['accuracy'] = task.info['accuracy']
    else:
        # something went wrong in the background job
        response = {
            'state': task.state,
            'current': 1,
            'total': 1,
            'status': str(task.info),  # this is the exception raised
        }
    return jsonify(response)


if __name__ == '__main__':
    app.run(debug=True)
