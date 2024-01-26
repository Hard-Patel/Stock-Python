from celery import Celery
from celery.schedules import crontab

celery = Celery(__name__, broker='redis://localhost:6379/0')


@celery.task
def update_model():
    # Implement logic to fetch real-time data and update the model
    pass

# Schedule the task to run every minute
celery.conf.beat_schedule = {
    'update-model-task': {
        'task': 'update_model',
        'schedule': crontab(minute='*/1'),
    },
}
