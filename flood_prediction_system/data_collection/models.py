from django.db import models

class RiverData(models.Model):
    date = models.DateField()
    river_discharge = models.FloatField()
    river_discharge_mean = models.FloatField()
    river_discharge_median = models.FloatField()
    river_discharge_max = models.FloatField()
    river_discharge_min = models.FloatField()
    river_discharge_p25 = models.FloatField()
    river_discharge_p75 = models.FloatField()

    def __str__(self):
        return f"River Data for {self.date}"

class WeatherData(models.Model):
    name = models.CharField(max_length=255)
    datetime = models.DateTimeField()
    tempmax = models.FloatField(null=True)
    tempmin = models.FloatField(null=True)
    temp = models.FloatField(null=True)
    feelslikemax = models.FloatField(null=True)
    feelslikemin = models.FloatField(null=True)
    feelslike = models.FloatField(null=True)
    dew = models.FloatField(null=True)
    humidity = models.FloatField(null=True)
    precip = models.FloatField(null=True)
    precipprob = models.FloatField(null=True)
    precipcover = models.FloatField(null=True)
    preciptype = models.CharField(max_length=255, null=True, blank=True)
    snow = models.FloatField(null=True)
    snowdepth = models.FloatField(null=True)
    windgust = models.FloatField(null=True)
    windspeed = models.FloatField(null=True)
    winddir = models.FloatField(null=True)
    pressure = models.FloatField(null=True)
    cloudcover = models.FloatField(null=True)
    visibility = models.FloatField(null=True)
    solarradiation = models.FloatField(null=True)
    solarenergy = models.FloatField(null=True)
    uvindex = models.FloatField(null=True)
    sunrise = models.CharField(max_length=10, null=True, blank=True)
    sunset = models.CharField(max_length=10, null=True, blank=True)
    moonphase = models.FloatField(null=True)
    conditions = models.CharField(max_length=255, null=True, blank=True)
    description = models.TextField(null=True, blank=True)
    icon = models.CharField(max_length=50, null=True, blank=True)
    stations = models.CharField(max_length=255, null=True, blank=True)

    def __str__(self):
        return f"Weather Data for {self.name} on {self.datetime}"