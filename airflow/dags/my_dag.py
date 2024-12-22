from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import yaml
import os
from tensorflow.keras.models import load_model
from train.train.run import train
from predict.predict.run import TextPredictionModel

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2021, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}



model_config_path = 'train/data/artefacts/2024-12-11-17-02-59/params.json'



def load_config(config_path):
    with open(config_path, 'r') as config_f:
        return yaml.safe_load(config_f)


def tester_load_model(artefacts_url):
    return load_model(os.path.join(artefacts_url, 'model.h5'))



def train_artefacts(**arguments):
    train_config = load_config(model_config_path)
    artefacts_url = arguments['artefacts_url']
    dataset_path = arguments['dataset_path']
    _, artefacts_url = train(dataset_path, train_config, artefacts_url)
    tester_load_model(artefacts_url)


def make_predictions(**arguments):
    artefacts_url = arguments['artefacts_url']
    text = arguments['text']
    top_k = arguments['top_k']
    model = TextPredictionModel.from_artefacts(artefacts_url)
    predictions = model.predict([text], top_k)
    print(f"Predictions for '{text}': {predictions}")





with DAG('my_dag', default_args=default_args, schedule_interval=timedelta(days=1)) as dag:

    train_model_task = PythonOperator(
        task_id='train_model',
        python_callable=train_artefacts,
        op_kwargs={'artefacts_url': 'train/data/artefacts',
                   'dataset_path': 'train/data/training-data/stackoverflow_posts.csv'},
        provide_context=True,
    )

    make_predictions_task = PythonOperator(
        task_id='make_predictions',
        python_callable=make_predictions,
        op_kwargs={'artefacts_url': 'train/data/artefacts',
                   'text': 'jaime bien apprendre le python',
                   'top_k': 5},
        provide_context=True,
    )

    train_model_task >> make_predictions_task
