# scheduler.py
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from django.conf import settings
from .tasks import fetch_river_data, fetch_weather_data

def start():
    if settings.SCHEDULER_ENABLED:  
        scheduler = BackgroundScheduler()
        
        scheduler.add_job(fetch_weather_data, CronTrigger(hour=0, minute=10), id="weather_fetch", replace_existing=True)
        scheduler.add_job(fetch_river_data, CronTrigger(hour=0, minute=10), id="river_fetch", replace_existing=True)

        scheduler.start()
        print("Scheduler started.")
